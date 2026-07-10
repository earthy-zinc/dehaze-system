package multilevel

import (
	"context"
	"errors"
	"math/rand"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/cache/errs"
	"github.com/earthyzinc/dehaze-go/pkg/cache/protection"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

// MultiLevelCache 多级缓存实现
// 读流程：L1(Local) -> L2(Redis) -> DataLoader
// 写流程：先写DB -> 删除Redis -> 删除Local（Cache-Aside Pattern）
type MultiLevelCache struct {
	opts *Options
}

// NewMultiLevelCache 创建多级缓存实例
func NewMultiLevelCache(opts ...Option) (*MultiLevelCache, error) {
	o := defaultOptions()
	for _, opt := range opts {
		opt(o)
	}

	if o.L1Cache == nil && o.L2Cache == nil {
		return nil, errors.New("at least one cache layer is required")
	}

	return &MultiLevelCache{opts: o}, nil
}

// Get 多级缓存读取
// 流程: L1 -> L2 -> (通过singleflight聚合) -> 返回并回填
func (m *MultiLevelCache) Get(ctx context.Context, key string) (string, error) {
	// 1. 布隆过滤器检查（防穿透）
	if m.opts.BloomFilter != nil && !m.opts.BloomFilter.MayExist(key) {
		return "", errs.ErrKeyNotFound
	}

	// 2. 尝试从L1获取
	if m.opts.L1Cache != nil {
		val, err := m.opts.L1Cache.Get(ctx, key)
		if err == nil {
			// 检查是否为空值标记
			if m.opts.NullCache != nil && m.opts.NullCache.IsNullValue(val) {
				return "", errs.ErrKeyNotFound
			}
			return val, nil
		}
	}

	// 3. 尝试从L2获取
	if m.opts.L2Cache != nil {
		val, err := m.getFromL2WithBreaker(ctx, key)
		if err == nil {
			// 检查是否为空值标记
			if m.opts.NullCache != nil && m.opts.NullCache.IsNullValue(val) {
				return "", errs.ErrKeyNotFound
			}
			// 回填L1
			m.writeBackToL1(ctx, key, val)
			return val, nil
		}
	}

	return "", errs.ErrKeyNotFound
}

// GetWithLoader 带数据加载器的多级缓存读取
// 当所有缓存层都miss时，使用loader加载数据
func (m *MultiLevelCache) GetWithLoader(ctx context.Context, key string, loader protection.DataLoader) (string, error) {
	// 1. 布隆过滤器检查（防穿透）
	if m.opts.BloomFilter != nil && !m.opts.BloomFilter.MayExist(key) {
		return "", errs.ErrKeyNotFound
	}

	// 2. 尝试从L1获取
	if m.opts.L1Cache != nil {
		val, err := m.opts.L1Cache.Get(ctx, key)
		if err == nil {
			if m.opts.NullCache != nil && m.opts.NullCache.IsNullValue(val) {
				return "", errs.ErrKeyNotFound
			}
			return val, nil
		}
	}

	// 3. 尝试从L2获取
	if m.opts.L2Cache != nil {
		val, err := m.getFromL2WithBreaker(ctx, key)
		if err == nil {
			if m.opts.NullCache != nil && m.opts.NullCache.IsNullValue(val) {
				return "", errs.ErrKeyNotFound
			}
			m.writeBackToL1(ctx, key, val)
			return val, nil
		}
	}

	// 4. 使用SingleFlight从数据源加载（防击穿）
	if loader != nil {
		return m.loadWithSingleFlight(ctx, key, loader)
	}

	return "", errs.ErrKeyNotFound
}

// Set 设置缓存
// 同时写入L1和L2
func (m *MultiLevelCache) Set(ctx context.Context, key string, value any, expiration time.Duration) error {
	// 添加随机过期时间（防雪崩）
	l1Exp := m.getExpireWithJitter(m.opts.L1DefaultExpire, expiration)
	l2Exp := m.getExpireWithJitter(m.opts.L2DefaultExpire, expiration)

	var errs []error

	// 先写L2
	if m.opts.L2Cache != nil {
		if err := m.opts.L2Cache.Set(ctx, key, value, l2Exp); err != nil {
			errs = append(errs, err)
			logger.Warn("写入L2缓存失败", zap.String("key", key), zap.Error(err))
		}
	}

	// 再写L1
	if m.opts.L1Cache != nil {
		if err := m.opts.L1Cache.Set(ctx, key, value, l1Exp); err != nil {
			errs = append(errs, err)
			logger.Warn("写入L1缓存失败", zap.String("key", key), zap.Error(err))
		}
	}

	// 更新布隆过滤器
	if m.opts.BloomFilter != nil {
		m.opts.BloomFilter.Add(key)
	}

	if len(errs) > 0 {
		return errs[0]
	}
	return nil
}

// Delete 删除缓存
// Cache-Aside模式：先删L2，再删L1
func (m *MultiLevelCache) Delete(ctx context.Context, keys ...string) error {
	var errs []error

	// 先删L2
	if m.opts.L2Cache != nil {
		if err := m.opts.L2Cache.Delete(ctx, keys...); err != nil {
			errs = append(errs, err)
			logger.Warn("删除L2缓存失败", zap.Strings("keys", keys), zap.Error(err))
		}
	}

	// 再删L1
	if m.opts.L1Cache != nil {
		if err := m.opts.L1Cache.Delete(ctx, keys...); err != nil {
			errs = append(errs, err)
			logger.Warn("删除L1缓存失败", zap.Strings("keys", keys), zap.Error(err))
		}
	}

	if len(errs) > 0 {
		return errs[0]
	}
	return nil
}

// Exists 检查key是否存在
func (m *MultiLevelCache) Exists(ctx context.Context, key string) (bool, error) {
	// 先查L1
	if m.opts.L1Cache != nil {
		if exists, _ := m.opts.L1Cache.Exists(ctx, key); exists {
			return true, nil
		}
	}

	// 再查L2
	if m.opts.L2Cache != nil {
		return m.opts.L2Cache.Exists(ctx, key)
	}

	return false, nil
}

// SetNX 仅当key不存在时设置
func (m *MultiLevelCache) SetNX(ctx context.Context, key string, value any, expiration time.Duration) (bool, error) {
	// 使用L2作为分布式锁的基础
	if m.opts.L2Cache != nil {
		ok, err := m.opts.L2Cache.SetNX(ctx, key, value, expiration)
		if err != nil || !ok {
			return ok, err
		}
	}

	// L2成功后设置L1
	if m.opts.L1Cache != nil {
		l1Exp := m.getExpireWithJitter(m.opts.L1DefaultExpire, expiration)
		_ = m.opts.L1Cache.Set(ctx, key, value, l1Exp)
	}

	return true, nil
}

// MGet 批量获取
func (m *MultiLevelCache) MGet(ctx context.Context, keys ...string) ([]string, error) {
	if len(keys) == 0 {
		return []string{}, nil
	}

	results := make([]string, len(keys))
	missedKeys := make([]string, 0)
	missedIndexes := make([]int, 0)

	// 先从L1批量获取
	if m.opts.L1Cache != nil {
		l1Results, err := m.opts.L1Cache.MGet(ctx, keys...)
		if err == nil {
			for i, val := range l1Results {
				if val != "" {
					results[i] = val
				} else {
					missedKeys = append(missedKeys, keys[i])
					missedIndexes = append(missedIndexes, i)
				}
			}
		} else {
			missedKeys = keys
			for i := range keys {
				missedIndexes = append(missedIndexes, i)
			}
		}
	} else {
		missedKeys = keys
		for i := range keys {
			missedIndexes = append(missedIndexes, i)
		}
	}

	// 从L2获取L1未命中的
	if len(missedKeys) > 0 && m.opts.L2Cache != nil {
		l2Results, err := m.opts.L2Cache.MGet(ctx, missedKeys...)
		if err == nil {
			for i, val := range l2Results {
				if val != "" {
					results[missedIndexes[i]] = val
					// 异步回填L1
					if m.opts.AsyncWriteBack && m.opts.L1Cache != nil {
						go func(k, v string) {
							_ = m.opts.L1Cache.Set(context.Background(), k, v, m.opts.L1DefaultExpire)
						}(missedKeys[i], val)
					}
				}
			}
		}
	}

	return results, nil
}

// MSet 批量设置
func (m *MultiLevelCache) MSet(ctx context.Context, kvPairs map[string]any, expiration time.Duration) error {
	l1Exp := m.getExpireWithJitter(m.opts.L1DefaultExpire, expiration)
	l2Exp := m.getExpireWithJitter(m.opts.L2DefaultExpire, expiration)

	// 先设置L2
	if m.opts.L2Cache != nil {
		if err := m.opts.L2Cache.MSet(ctx, kvPairs, l2Exp); err != nil {
			return err
		}
	}

	// 再设置L1
	if m.opts.L1Cache != nil {
		_ = m.opts.L1Cache.MSet(ctx, kvPairs, l1Exp)
	}

	// 更新布隆过滤器
	if m.opts.BloomFilter != nil {
		for key := range kvPairs {
			m.opts.BloomFilter.Add(key)
		}
	}

	return nil
}

// MDelete 批量删除
func (m *MultiLevelCache) MDelete(ctx context.Context, keys ...string) error {
	return m.Delete(ctx, keys...)
}

// Incr 自增
func (m *MultiLevelCache) Incr(ctx context.Context, key string) (int64, error) {
	// 计数器操作使用L2保证一致性
	if m.opts.L2Cache != nil {
		val, err := m.opts.L2Cache.Incr(ctx, key)
		if err != nil {
			return 0, err
		}
		// 删除L1缓存，保证一致性
		if m.opts.L1Cache != nil {
			_ = m.opts.L1Cache.Delete(ctx, key)
		}
		return val, nil
	}

	if m.opts.L1Cache != nil {
		return m.opts.L1Cache.Incr(ctx, key)
	}

	return 0, errors.New("no cache available")
}

// IncrBy 增加指定值
func (m *MultiLevelCache) IncrBy(ctx context.Context, key string, value int64) (int64, error) {
	if m.opts.L2Cache != nil {
		val, err := m.opts.L2Cache.IncrBy(ctx, key, value)
		if err != nil {
			return 0, err
		}
		if m.opts.L1Cache != nil {
			_ = m.opts.L1Cache.Delete(ctx, key)
		}
		return val, nil
	}

	if m.opts.L1Cache != nil {
		return m.opts.L1Cache.IncrBy(ctx, key, value)
	}

	return 0, errors.New("no cache available")
}

// Decr 自减
func (m *MultiLevelCache) Decr(ctx context.Context, key string) (int64, error) {
	if m.opts.L2Cache != nil {
		val, err := m.opts.L2Cache.Decr(ctx, key)
		if err != nil {
			return 0, err
		}
		if m.opts.L1Cache != nil {
			_ = m.opts.L1Cache.Delete(ctx, key)
		}
		return val, nil
	}

	if m.opts.L1Cache != nil {
		return m.opts.L1Cache.Decr(ctx, key)
	}

	return 0, errors.New("no cache available")
}

// DecrBy 减少指定值
func (m *MultiLevelCache) DecrBy(ctx context.Context, key string, value int64) (int64, error) {
	if m.opts.L2Cache != nil {
		val, err := m.opts.L2Cache.DecrBy(ctx, key, value)
		if err != nil {
			return 0, err
		}
		if m.opts.L1Cache != nil {
			_ = m.opts.L1Cache.Delete(ctx, key)
		}
		return val, nil
	}

	if m.opts.L1Cache != nil {
		return m.opts.L1Cache.DecrBy(ctx, key, value)
	}

	return 0, errors.New("no cache available")
}

// Expire 设置过期时间
func (m *MultiLevelCache) Expire(ctx context.Context, key string, expiration time.Duration) (bool, error) {
	var success bool

	if m.opts.L2Cache != nil {
		ok, err := m.opts.L2Cache.Expire(ctx, key, expiration)
		if err != nil {
			return false, err
		}
		success = ok
	}

	if m.opts.L1Cache != nil {
		l1Exp := m.getExpireWithJitter(m.opts.L1DefaultExpire, expiration)
		_, _ = m.opts.L1Cache.Expire(ctx, key, l1Exp)
	}

	return success, nil
}

// TTL 获取剩余过期时间
func (m *MultiLevelCache) TTL(ctx context.Context, key string) (time.Duration, error) {
	// 优先返回L2的TTL
	if m.opts.L2Cache != nil {
		return m.opts.L2Cache.TTL(ctx, key)
	}

	if m.opts.L1Cache != nil {
		return m.opts.L1Cache.TTL(ctx, key)
	}

	return -2 * time.Second, nil
}

// Lock 分布式锁
func (m *MultiLevelCache) Lock(ctx context.Context, key string, expiration time.Duration) (bool, error) {
	// 分布式锁必须使用L2
	if m.opts.L2Cache != nil {
		return m.opts.L2Cache.Lock(ctx, key, expiration)
	}

	if m.opts.L1Cache != nil {
		return m.opts.L1Cache.Lock(ctx, key, expiration)
	}

	return false, errors.New("no cache available for lock")
}

// Unlock 释放分布式锁
func (m *MultiLevelCache) Unlock(ctx context.Context, key string) (bool, error) {
	if m.opts.L2Cache != nil {
		return m.opts.L2Cache.Unlock(ctx, key)
	}

	if m.opts.L1Cache != nil {
		return m.opts.L1Cache.Unlock(ctx, key)
	}

	return false, errors.New("no cache available for unlock")
}

// Pipeline 批量操作
func (m *MultiLevelCache) Pipeline(ctx context.Context, ops []types.PipelineOp) error {
	// Pipeline操作直接使用L2
	if m.opts.L2Cache != nil {
		err := m.opts.L2Cache.Pipeline(ctx, ops)
		if err != nil {
			return err
		}
		// 清理L1中对应的key
		if m.opts.L1Cache != nil {
			for _, op := range ops {
				if op.Type == "set" || op.Type == "delete" {
					_ = m.opts.L1Cache.Delete(ctx, op.Key)
				}
			}
		}
		return nil
	}

	if m.opts.L1Cache != nil {
		return m.opts.L1Cache.Pipeline(ctx, ops)
	}

	return errors.New("no cache available")
}

// ========== Hash 操作 ==========

// HGet 获取哈希表中指定字段的值
func (m *MultiLevelCache) HGet(ctx context.Context, key, field string) (string, error) {
	// 先从L1获取
	if m.opts.L1Cache != nil {
		val, err := m.opts.L1Cache.HGet(ctx, key, field)
		if err == nil {
			return val, nil
		}
	}

	// 从L2获取
	if m.opts.L2Cache != nil {
		val, err := m.opts.L2Cache.HGet(ctx, key, field)
		if err == nil {
			// 回填L1
			if m.opts.L1Cache != nil && m.opts.AsyncWriteBack {
				go func() {
					_ = m.opts.L1Cache.HSet(context.Background(), key, field, val)
				}()
			} else if m.opts.L1Cache != nil {
				_ = m.opts.L1Cache.HSet(ctx, key, field, val)
			}
			return val, nil
		}
		return "", err
	}

	return "", errs.ErrKeyNotFound
}

// HSet 设置哈希表中指定字段的值
func (m *MultiLevelCache) HSet(ctx context.Context, key, field string, value any) error {
	// 先设置L2
	if m.opts.L2Cache != nil {
		if err := m.opts.L2Cache.HSet(ctx, key, field, value); err != nil {
			return err
		}
	}

	// 删除L1中的缓存（保持一致性）
	if m.opts.L1Cache != nil {
		_ = m.opts.L1Cache.HDel(ctx, key, field)
	}

	return nil
}

// HDel 删除哈希表中的一个或多个字段
func (m *MultiLevelCache) HDel(ctx context.Context, key string, fields ...string) error {
	if len(fields) == 0 {
		return nil
	}

	// 先删L2
	if m.opts.L2Cache != nil {
		if err := m.opts.L2Cache.HDel(ctx, key, fields...); err != nil {
			return err
		}
	}

	// 再删L1
	if m.opts.L1Cache != nil {
		_ = m.opts.L1Cache.HDel(ctx, key, fields...)
	}

	return nil
}

// HGetAll 获取哈希表中所有字段和值
func (m *MultiLevelCache) HGetAll(ctx context.Context, key string) (map[string]string, error) {
	// HGetAll 直接从L2获取（保证数据完整性）
	if m.opts.L2Cache != nil {
		return m.opts.L2Cache.HGetAll(ctx, key)
	}

	if m.opts.L1Cache != nil {
		return m.opts.L1Cache.HGetAll(ctx, key)
	}

	return make(map[string]string), nil
}

// ========== Set 操作 ==========

// SAdd 向集合添加一个或多个成员
func (m *MultiLevelCache) SAdd(ctx context.Context, key string, members ...any) error {
	if len(members) == 0 {
		return nil
	}

	// 先添加到L2
	if m.opts.L2Cache != nil {
		if err := m.opts.L2Cache.SAdd(ctx, key, members...); err != nil {
			return err
		}
	}

	// 删除L1中的集合缓存（保持一致性）
	if m.opts.L1Cache != nil {
		_ = m.opts.L1Cache.Delete(ctx, key)
	}

	return nil
}

// SMembers 返回集合中的所有成员
func (m *MultiLevelCache) SMembers(ctx context.Context, key string) ([]string, error) {
	// SMembers 直接从L2获取（保证数据完整性）
	if m.opts.L2Cache != nil {
		return m.opts.L2Cache.SMembers(ctx, key)
	}

	if m.opts.L1Cache != nil {
		return m.opts.L1Cache.SMembers(ctx, key)
	}

	return []string{}, nil
}

// SRem 移除集合中一个或多个成员
func (m *MultiLevelCache) SRem(ctx context.Context, key string, members ...any) error {
	if len(members) == 0 {
		return nil
	}

	// 先从L2移除
	if m.opts.L2Cache != nil {
		if err := m.opts.L2Cache.SRem(ctx, key, members...); err != nil {
			return err
		}
	}

	// 删除L1中的集合缓存（保持一致性）
	if m.opts.L1Cache != nil {
		_ = m.opts.L1Cache.Delete(ctx, key)
	}

	return nil
}

// ========== 内部方法 ==========

// getFromL2WithBreaker 通过熔断器从L2获取数据
func (m *MultiLevelCache) getFromL2WithBreaker(ctx context.Context, key string) (string, error) {
	if m.opts.Breaker == nil {
		return m.opts.L2Cache.Get(ctx, key)
	}

	var val string
	err := m.opts.Breaker.Execute(func() error {
		var getErr error
		val, getErr = m.opts.L2Cache.Get(ctx, key)
		return getErr
	})

	if errors.Is(err, protection.ErrCircuitOpen) {
		logger.Warn("L2缓存熔断器打开，跳过L2读取", zap.String("key", key))
		return "", err
	}

	return val, err
}

// writeBackToL1 回写L1缓存
func (m *MultiLevelCache) writeBackToL1(ctx context.Context, key string, value string) {
	if m.opts.L1Cache == nil {
		return
	}

	l1Exp := m.getExpireWithJitter(m.opts.L1DefaultExpire, 0)

	if m.opts.AsyncWriteBack {
		go func() {
			if err := m.opts.L1Cache.Set(context.Background(), key, value, l1Exp); err != nil {
				logger.Warn("异步回写L1失败", zap.String("key", key), zap.Error(err))
			}
		}()
	} else {
		if err := m.opts.L1Cache.Set(ctx, key, value, l1Exp); err != nil {
			logger.Warn("回写L1失败", zap.String("key", key), zap.Error(err))
		}
	}
}

// loadWithSingleFlight 使用SingleFlight从数据源加载
func (m *MultiLevelCache) loadWithSingleFlight(ctx context.Context, key string, loader protection.DataLoader) (string, error) {
	var result any
	var err error

	if m.opts.SingleFlight != nil {
		result, err = m.opts.SingleFlight.Do(ctx, key, func() (any, error) {
			return loader(ctx, key)
		})
	} else {
		result, err = loader(ctx, key)
	}

	if err != nil {
		// 数据不存在，设置空值缓存（防穿透）
		if m.opts.NullCache != nil && errors.Is(err, errs.ErrKeyNotFound) {
			_ = m.opts.NullCache.SetNull(ctx, key, 0)
		}
		return "", err
	}

	val, ok := result.(string)
	if !ok {
		return "", errors.New("invalid data type from loader")
	}

	// 将数据写入缓存
	_ = m.Set(ctx, key, val, 0)

	return val, nil
}

// getExpireWithJitter 获取带随机抖动的过期时间（防雪崩）
func (m *MultiLevelCache) getExpireWithJitter(defaultExp, customExp time.Duration) time.Duration {
	exp := defaultExp
	if customExp > 0 {
		exp = customExp
	}

	if m.opts.RandomExpireRange > 0 {
		jitter := time.Duration(rand.Int63n(int64(m.opts.RandomExpireRange)))
		exp += jitter
	}

	return exp
}

// SetNullValue 手动设置空值缓存
func (m *MultiLevelCache) SetNullValue(ctx context.Context, key string) error {
	if m.opts.NullCache != nil {
		return m.opts.NullCache.SetNull(ctx, key, 0)
	}
	return nil
}

// AddToBloomFilter 添加key到布隆过滤器
func (m *MultiLevelCache) AddToBloomFilter(keys ...string) {
	if m.opts.BloomFilter != nil {
		for _, key := range keys {
			m.opts.BloomFilter.Add(key)
		}
	}
}

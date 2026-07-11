package local

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/cache/errs"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/google/uuid"
	"github.com/songzhibin97/gkit/cache/local_cache"
)

type LocalCache struct {
	cache    local_cache.Cache
	counters sync.Map // 用于增量操作的计数器存储
	incrMu   sync.Map // per-key mutex for atomic IncrBy/DecrBy
	ttlMap   sync.Map // 用于存储TTL信息
	locks    sync.Map // 用于分布式锁
	hashData sync.Map // 用于Hash数据结构存储
	setData  sync.Map // 用于Set数据结构存储
}

func NewLocalCache(localCache local_cache.Cache) *LocalCache {
	return &LocalCache{
		cache: localCache,
	}
}

func (c *LocalCache) Get(ctx context.Context, key string) (string, error) {
	val, found := c.cache.Get(key)
	if !found {
		return "", errs.ErrKeyNotFound
	}
	if strVal, ok := val.(string); ok {
		return strVal, nil
	}
	return fmt.Sprintf("%v", val), nil
}

func (c *LocalCache) Set(ctx context.Context, key string, value any, expiration time.Duration) error {
	if expiration > 0 {
		c.cache.Set(key, value, expiration)
		c.ttlMap.Store(key, time.Now().Add(expiration))
	} else {
		c.cache.SetDefault(key, value)
	}
	return nil
}

func (c *LocalCache) Delete(ctx context.Context, keys ...string) error {
	for _, key := range keys {
		c.cache.Delete(key)
		c.counters.Delete(key)
		c.ttlMap.Delete(key)
		c.locks.Delete(key)
	}
	return nil
}

func (c *LocalCache) Exists(ctx context.Context, key string) (bool, error) {
	_, found := c.cache.Get(key)
	return found, nil
}

func (c *LocalCache) SetNX(ctx context.Context, key string, value any, expiration time.Duration) (bool, error) {
	if _, found := c.cache.Get(key); found {
		return false, nil
	}
	c.cache.Set(key, value, expiration)
	return true, nil
}

// ========== 批量操作 ==========

func (c *LocalCache) MGet(ctx context.Context, keys ...string) ([]string, error) {
	if len(keys) == 0 {
		return []string{}, nil
	}
	result := make([]string, len(keys))
	for i, key := range keys {
		if val, found := c.cache.Get(key); found {
			if strVal, ok := val.(string); ok {
				result[i] = strVal
			} else {
				result[i] = fmt.Sprintf("%v", val)
			}
		} else {
			result[i] = ""
		}
	}
	return result, nil
}

func (c *LocalCache) MSet(ctx context.Context, kvPairs map[string]any, expiration time.Duration) error {
	if len(kvPairs) == 0 {
		return nil
	}
	for key, value := range kvPairs {
		c.cache.Set(key, value, expiration)
		if expiration > 0 {
			c.ttlMap.Store(key, time.Now().Add(expiration))
		}
	}
	return nil
}

func (c *LocalCache) MDelete(ctx context.Context, keys ...string) error {
	return c.Delete(ctx, keys...)
}

// ========== 增量操作 ==========

func (c *LocalCache) Incr(ctx context.Context, key string) (int64, error) {
	return c.IncrBy(ctx, key, 1)
}

func (c *LocalCache) IncrBy(ctx context.Context, key string, value int64) (int64, error) {
	// 使用 per-key mutex 保证原子性
	muVal, _ := c.incrMu.LoadOrStore(key, &sync.Mutex{})
	mu := muVal.(*sync.Mutex)
	mu.Lock()
	defer mu.Unlock()

	var newValue int64
	if val, found := c.cache.Get(key); found {
		switch v := val.(type) {
		case int:
			newValue = int64(v) + value
		case int64:
			newValue = v + value
		case int32:
			newValue = int64(v) + value
		case float64:
			newValue = int64(v) + value
		case string:
			var parsed int64
			if _, err := fmt.Sscanf(v, "%d", &parsed); err != nil {
				return 0, fmt.Errorf("value is not a valid integer")
			}
			newValue = parsed + value
		default:
			newValue = value
		}
	} else {
		newValue = value
	}

	c.cache.Set(key, newValue, 0)
	c.counters.Store(key, newValue)
	return newValue, nil
}

func (c *LocalCache) Decr(ctx context.Context, key string) (int64, error) {
	return c.DecrBy(ctx, key, 1)
}

func (c *LocalCache) DecrBy(ctx context.Context, key string, value int64) (int64, error) {
	return c.IncrBy(ctx, key, -value)
}

// ========== TTL管理 ==========

func (c *LocalCache) Expire(ctx context.Context, key string, expiration time.Duration) (bool, error) {
	if _, found := c.cache.Get(key); !found {
		return false, nil
	}
	// 本地缓存的TTL管理是软性的，我们记录过期时间
	c.ttlMap.Store(key, time.Now().Add(expiration))
	return true, nil
}

func (c *LocalCache) TTL(ctx context.Context, key string) (time.Duration, error) {
	if _, found := c.cache.Get(key); !found {
		return -2 * time.Second, nil // key不存在
	}
	if exp, found := c.ttlMap.Load(key); found {
		if expTime, ok := exp.(time.Time); ok {
			remaining := time.Until(expTime)
			if remaining <= 0 {
				return -1 * time.Second, nil // 已过期
			}
			return remaining, nil
		}
	}
	return -1 * time.Second, nil // 没有设置过期时间
}

// ========== 分布式锁 ==========

const (
	defaultLockPrefix = "lock:"
)

func (c *LocalCache) Lock(ctx context.Context, key string, expiration time.Duration) (string, bool, error) {
	lockKey := fmt.Sprintf("%s%s", defaultLockPrefix, key)
	token := uuid.New().String()
	if _, found := c.locks.Load(lockKey); found {
		return "", false, nil
	}
	c.locks.Store(lockKey, token)
	return token, true, nil
}

func (c *LocalCache) Unlock(ctx context.Context, key string, token string) (bool, error) {
	lockKey := fmt.Sprintf("%s%s", defaultLockPrefix, key)
	existing, found := c.locks.Load(lockKey)
	if !found {
		return false, nil
	}
	// 仅当 token 匹配时才释放
	if existing.(string) != token {
		return false, nil
	}
	c.locks.Delete(lockKey)
	return true, nil
}

// ========== Pipeline/事务 ==========

func (c *LocalCache) Pipeline(ctx context.Context, ops []types.PipelineOp) error {
	if len(ops) == 0 {
		return nil
	}

	// 本地缓存的Pipeline实现为顺序执行（本地操作本身就是原子的）
	for _, op := range ops {
		switch op.Type {
		case "set":
			if err := c.Set(ctx, op.Key, op.Value, op.Exp); err != nil {
				return err
			}
		case "delete":
			if err := c.Delete(ctx, op.Key); err != nil {
				return err
			}
		case "incr":
			if _, err := c.Incr(ctx, op.Key); err != nil {
				return err
			}
		case "decr":
			if _, err := c.Decr(ctx, op.Key); err != nil {
				return err
			}
		}
	}
	return nil
}

// ========== Hash 操作 ==========

// HGet 获取哈希表中指定字段的值
func (c *LocalCache) HGet(ctx context.Context, key, field string) (string, error) {
	val, ok := c.hashData.Load(key)
	if !ok {
		return "", errs.ErrKeyNotFound
	}

	hashMap, ok := val.(*sync.Map)
	if !ok {
		return "", errs.ErrKeyNotFound
	}

	fieldVal, ok := hashMap.Load(field)
	if !ok {
		return "", errs.ErrKeyNotFound
	}

	if strVal, ok := fieldVal.(string); ok {
		return strVal, nil
	}
	return fmt.Sprintf("%v", fieldVal), nil
}

// HSet 设置哈希表中指定字段的值
func (c *LocalCache) HSet(ctx context.Context, key, field string, value any) error {
	var hashMap *sync.Map

	val, ok := c.hashData.Load(key)
	if ok {
		hashMap, _ = val.(*sync.Map)
	}
	if hashMap == nil {
		hashMap = &sync.Map{}
		c.hashData.Store(key, hashMap)
	}

	hashMap.Store(field, value)
	return nil
}

// HDel 删除哈希表中的一个或多个字段
func (c *LocalCache) HDel(ctx context.Context, key string, fields ...string) error {
	if len(fields) == 0 {
		return nil
	}

	val, ok := c.hashData.Load(key)
	if !ok {
		return nil
	}

	hashMap, ok := val.(*sync.Map)
	if !ok {
		return nil
	}

	for _, field := range fields {
		hashMap.Delete(field)
	}
	return nil
}

// HGetAll 获取哈希表中所有字段和值
func (c *LocalCache) HGetAll(ctx context.Context, key string) (map[string]string, error) {
	result := make(map[string]string)

	val, ok := c.hashData.Load(key)
	if !ok {
		return result, nil
	}

	hashMap, ok := val.(*sync.Map)
	if !ok {
		return result, nil
	}

	hashMap.Range(func(k, v any) bool {
		keyStr, _ := k.(string)
		if valStr, ok := v.(string); ok {
			result[keyStr] = valStr
		} else {
			result[keyStr] = fmt.Sprintf("%v", v)
		}
		return true
	})

	return result, nil
}

// ========== Set 操作 ==========

// SAdd 向集合添加一个或多个成员
func (c *LocalCache) SAdd(ctx context.Context, key string, members ...any) error {
	if len(members) == 0 {
		return nil
	}

	var setMap *sync.Map

	val, ok := c.setData.Load(key)
	if ok {
		setMap, _ = val.(*sync.Map)
	}
	if setMap == nil {
		setMap = &sync.Map{}
		c.setData.Store(key, setMap)
	}

	for _, member := range members {
		memberStr := fmt.Sprintf("%v", member)
		setMap.Store(memberStr, struct{}{})
	}
	return nil
}

// SMembers 返回集合中的所有成员
func (c *LocalCache) SMembers(ctx context.Context, key string) ([]string, error) {
	result := make([]string, 0)

	val, ok := c.setData.Load(key)
	if !ok {
		return result, nil
	}

	setMap, ok := val.(*sync.Map)
	if !ok {
		return result, nil
	}

	setMap.Range(func(k, v any) bool {
		if keyStr, ok := k.(string); ok {
			result = append(result, keyStr)
		}
		return true
	})

	return result, nil
}

// SRem 移除集合中一个或多个成员
func (c *LocalCache) SRem(ctx context.Context, key string, members ...any) error {
	if len(members) == 0 {
		return nil
	}

	val, ok := c.setData.Load(key)
	if !ok {
		return nil
	}

	setMap, ok := val.(*sync.Map)
	if !ok {
		return nil
	}

	for _, member := range members {
		memberStr := fmt.Sprintf("%v", member)
		setMap.Delete(memberStr)
	}
	return nil
}

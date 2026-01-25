package cache

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/cache/local"
	"github.com/earthyzinc/dehaze-go/pkg/cache/multilevel"
	"github.com/earthyzinc/dehaze-go/pkg/cache/protection"
	"github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

var (
	cacheManager *CacheManager
	once         sync.Once
)

type CacheManager struct {
	config *options.Cache

	localCache *local.LocalCache
	redisCache *redis.RedisCache

	// 多级缓存
	multiLevelCache *multilevel.MultiLevelCache

	// 防护组件
	bloomFilter  *protection.BloomFilter
	singleFlight *protection.SingleFlight
	breaker      *protection.Breaker
	nullCache    *protection.NullCache

	mu sync.RWMutex
}

func Init() *CacheManager {
	once.Do(func() {
		cacheManager = &CacheManager{
			config: &config.GetConfig().Cache,
		}

		if cacheManager.config.Local.Enabled {
			cacheManager.localCache = local.InitLocalCache()
		}

		if cacheManager.config.Redis.Enabled {
			cacheManager.redisCache, _ = redis.InitRedis()
		}

		if cacheManager.localCache == nil && cacheManager.redisCache == nil {
			logger.Error("所有缓存后端都不可用")
			return
		}

		// 初始化防护组件
		cacheManager.initProtection()

		// 初始化多级缓存
		if cacheManager.config.MultiLevel.Enabled {
			cacheManager.initMultiLevelCache()
		}
		logger.Info("缓存管理器初始化成功")
	})

	return cacheManager
}

func GetCacheManager() *CacheManager {
	if cacheManager == nil {
		return Init()
	}
	return cacheManager
}

func GetCache() types.ICache {
	manager := GetCacheManager()
	if manager == nil {
		return nil
	}
	return manager.GetCache()
}

// initProtection 初始化防护组件
func (m *CacheManager) initProtection() {
	protCfg := m.config.Protection

	// 初始化布隆过滤器
	if protCfg.BloomFilter.Enabled {
		m.bloomFilter = protection.NewBloomFilter(
			protCfg.BloomFilter.ExpectedItems,
			protCfg.BloomFilter.FalsePositiveRate,
		)
		logger.Info("布隆过滤器初始化成功")
	}

	// 初始化单飞
	if protCfg.SingleFlight.Enabled {
		m.singleFlight = protection.NewSingleFlight()
		logger.Info("SingleFlight初始化成功")
	}

	// 初始化熔断器
	if protCfg.CircuitBreaker.Enabled {
		m.breaker = protection.NewBreaker(
			protection.WithFailureThreshold(protCfg.CircuitBreaker.FailureThreshold),
			protection.WithTimeout(time.Duration(protCfg.CircuitBreaker.Timeout)*time.Second),
			protection.WithMaxRequests(uint(protCfg.CircuitBreaker.MaxRequests)),
		)
		logger.Info("熔断器初始化成功")
	}

	// 初始化空值缓存
	if protCfg.NullCache.Enabled {
		var baseCache types.ICache
		if m.localCache != nil {
			baseCache = m.localCache
		} else if m.redisCache != nil {
			baseCache = m.redisCache
		}
		if baseCache != nil {
			m.nullCache = protection.NewNullCache(baseCache, protCfg.NullCache.ExpireSeconds)
			logger.Info("空值缓存初始化成功")
		}
	}
}

// initMultiLevelCache 初始化多级缓存
func (m *CacheManager) initMultiLevelCache() {
	mlCfg := m.config.MultiLevel

	opts := []multilevel.Option{
		multilevel.WithL1DefaultExpire(time.Duration(mlCfg.L1ExpireSeconds) * time.Second),
		multilevel.WithL2DefaultExpire(time.Duration(mlCfg.L2ExpireSeconds) * time.Second),
		multilevel.WithRandomExpireRange(time.Duration(mlCfg.RandomExpireRange) * time.Second),
		multilevel.WithAsyncWriteBack(mlCfg.AsyncWriteBack),
	}

	if m.localCache != nil {
		opts = append(opts, multilevel.WithL1Cache(m.localCache))
	}
	if m.redisCache != nil {
		opts = append(opts, multilevel.WithL2Cache(m.redisCache))
	}
	if m.bloomFilter != nil {
		opts = append(opts, multilevel.WithBloomFilter(m.bloomFilter))
	}
	if m.singleFlight != nil {
		opts = append(opts, multilevel.WithSingleFlight(m.singleFlight))
	}
	if m.breaker != nil {
		opts = append(opts, multilevel.WithBreaker(m.breaker))
	}
	if m.nullCache != nil {
		opts = append(opts, multilevel.WithNullCache(m.nullCache))
	}

	var err error
	m.multiLevelCache, err = multilevel.NewMultiLevelCache(opts...)
	if err != nil {
		logger.Error("多级缓存初始化失败", zap.Error(err))
		return
	}

	logger.Info("多级缓存初始化成功")
}

func (m *CacheManager) GetCache() types.ICache {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// 优先使用多级缓存
	if m.multiLevelCache != nil {
		return m.multiLevelCache
	}

	if m.redisCache != nil {
		return m.redisCache
	}

	if cacheManager.config.Fallback.Enabled && m.localCache != nil {
		logger.Warn("Redis 不可用，使用本地缓存")
		return m.localCache
	}

	logger.Error("所有缓存后端都不可用")
	return nil
}

// GetMultiLevelCache 获取多级缓存实例
func (m *CacheManager) GetMultiLevelCache() *multilevel.MultiLevelCache {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.multiLevelCache
}

// GetL1Cache 获取L1缓存（本地缓存）
func (m *CacheManager) GetL1Cache() types.ICache {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.localCache
}

// GetL2Cache 获取L2缓存（Redis缓存）
func (m *CacheManager) GetL2Cache() types.ICache {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.redisCache
}

// GetBloomFilter 获取布隆过滤器
func (m *CacheManager) GetBloomFilter() *protection.BloomFilter {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.bloomFilter
}

// HealthCheck 健康检查
func (m *CacheManager) HealthCheck(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.redisCache == nil {
		logger.Warn("Redis 未初始化")
		return nil
	}

	redisClient := redis.GetClient()
	if err := redisClient.Ping(ctx).Err(); err != nil {
		logger.Warn("Redis 健康检查失败", zap.Error(err))
		if cacheManager.config.Fallback.Enabled {
			return nil // 降级情况下不返回错误
		}
		return fmt.Errorf("redis health check failed: %w", err)
	}

	return nil
}

func (m *CacheManager) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var errs []error

	if m.redisCache != nil {
		if err := redis.Close(); err != nil {
			logger.Error("关闭 Redis 连接失败", zap.Error(err))
			errs = append(errs, err)
		}
	}

	if m.localCache != nil {
		logger.Info("本地缓存已关闭")
	}

	if len(errs) > 0 {
		return fmt.Errorf("关闭缓存时发生 %d 个错误", len(errs))
	}

	logger.Info("缓存管理器已关闭")
	return nil
}

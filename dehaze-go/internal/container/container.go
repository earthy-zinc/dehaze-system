package container

import (
	"sync"

	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

// Container 应用级依赖注入容器
// 遵循依赖倒置原则，上层模块依赖接口而非具体实现
type Container struct {
	db     *gorm.DB
	redis  *redis.Client
	logger *zap.Logger
	config *options.AsyncTask
	mu     sync.RWMutex

	// Repository 实例缓存
	repositories map[string]any

	// Service 实例缓存
	services map[string]any

	// Cache 实例缓存
	caches map[string]any
}

var (
	instance *Container
	once     sync.Once
)

// New 创建新的容器实例（用于测试场景）
func New(db *gorm.DB, redis *redis.Client, logger *zap.Logger) *Container {
	return &Container{
		db:           db,
		redis:        redis,
		logger:       logger,
		repositories: make(map[string]any),
		services:     make(map[string]any),
		caches:       make(map[string]any),
	}
}

// NewWithConfig 创建带配置的容器实例
func NewWithConfig(db *gorm.DB, redis *redis.Client, logger *zap.Logger, cfg *options.AsyncTask) *Container {
	return &Container{
		db:           db,
		redis:        redis,
		logger:       logger,
		config:       cfg,
		repositories: make(map[string]any),
		services:     make(map[string]any),
		caches:       make(map[string]any),
	}
}

// Init 初始化全局容器单例
func Init(db *gorm.DB, redis *redis.Client, logger *zap.Logger) {
	once.Do(func() {
		instance = New(db, redis, logger)
	})
}

// InitWithConfig 初始化全局容器单例（带配置）
func InitWithConfig(db *gorm.DB, redis *redis.Client, logger *zap.Logger, cfg *options.AsyncTask) {
	once.Do(func() {
		instance = NewWithConfig(db, redis, logger, cfg)
	})
}

// GetInstance 获取全局容器实例
func GetInstance() *Container {
	if instance == nil {
		panic("container not initialized, call Init() first")
	}
	return instance
}

// DB 获取数据库连接
func (c *Container) DB() *gorm.DB {
	return c.db
}

// Redis 获取 Redis 客户端
func (c *Container) Redis() *redis.Client {
	return c.redis
}

// Logger 获取日志实例
func (c *Container) Logger() *zap.Logger {
	return c.logger
}

// Config 获取异步任务配置
func (c *Container) Config() *options.AsyncTask {
	return c.config
}

// SetConfig 设置异步任务配置
func (c *Container) SetConfig(cfg *options.AsyncTask) {
	c.config = cfg
}

// Register 注册组件到容器
func (c *Container) Register(category, name string, component any) {
	c.mu.Lock()
	defer c.mu.Unlock()

	switch category {
	case "repository":
		c.repositories[name] = component
	case "service":
		c.services[name] = component
	case "cache":
		c.caches[name] = component
	}
}

// GetRepository 获取 Repository 实例
func (c *Container) GetRepository(name string) any {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.repositories[name]
}

// GetService 获取 Service 实例
func (c *Container) GetService(name string) any {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.services[name]
}

// GetCache 获取 Cache 实例
func (c *Container) GetCache(name string) any {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.caches[name]
}

// MustGetRepository 获取 Repository，不存在则 panic
func MustGetRepository[T any](c *Container, name string) T {
	repo := c.GetRepository(name)
	if repo == nil {
		panic("repository not found: " + name)
	}
	return repo.(T)
}

// MustGetService 获取 Service，不存在则 panic
func MustGetService[T any](c *Container, name string) T {
	svc := c.GetService(name)
	if svc == nil {
		panic("service not found: " + name)
	}
	return svc.(T)
}

// MustGetCache 获取 Cache，不存在则 panic
func MustGetCache[T any](c *Container, name string) T {
	cache := c.GetCache(name)
	if cache == nil {
		panic("cache not found: " + name)
	}
	return cache.(T)
}

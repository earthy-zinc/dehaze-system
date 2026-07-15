package redis

import (
	"context"
	"fmt"
	"sync"

	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

var (
	_client  *redis.Client
	_once    sync.Once
	_initErr error
)

func InitRedis() (*RedisCache, error) {
	cfg := config.GetConfig().Cache.Redis

	_once.Do(func() {
		poolSize := cfg.PoolSize
		if poolSize <= 0 {
			poolSize = 10
		}
		opts := &redis.Options{
			Addr:     cfg.Addr,
			Password: cfg.Password,
			DB:       cfg.DB,
			PoolSize: poolSize,
		}

		_client = redis.NewClient(opts)

		if err := _client.Ping(context.Background()).Err(); err != nil {
			_initErr = fmt.Errorf("Redis连接失败: %w", err)
			_client = nil
			logger.Error("Redis连接初始化失败", zap.Error(err))
			return
		}
	})

	// 初始化失败时返回 nil client，由调用方处理错误，不返回包装了 nil 的无效实例
	if _initErr != nil {
		return nil, _initErr
	}
	return NewRedisCache(_client), nil
}

func GetClient() *redis.Client {
	return _client
}

func Close() error {
	if _client == nil {
		return nil
	}
	if err := _client.Close(); err != nil {
		logger.Error("关闭Redis连接失败", zap.Error(err))
		return err
	}
	logger.Info("Redis连接已关闭")
	return nil
}

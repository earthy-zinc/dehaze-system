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
	_client *redis.Client
	_once   sync.Once
)

func InitRedis() (*RedisCache, error) {
	var initErr error
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
			initErr = fmt.Errorf("Redis连接失败: %w", err)
			_client = nil
			logger.Error("Redis连接初始化失败", zap.Error(err))
			return
		}
	})

	return NewRedisCache(_client), initErr
}

func GetClient() *redis.Client {
	return _client
}

func Close() error {
	if err := _client.Close(); err != nil {
		logger.Error("关闭Redis连接失败", zap.Error(err))
		return err
	}
	logger.Info("Redis连接已关闭")
	return nil
}

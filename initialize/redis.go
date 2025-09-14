package initialize

import (
	"context"

	"github.com/earthyzinc/dehaze-go/config"
	"github.com/earthyzinc/dehaze-go/global"

	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

func initRedisClient(redisCfg config.Redis) (redis.UniversalClient, error) {
	if redisCfg.Addr == "" && len(redisCfg.ClusterAddrs) == 0 {
		global.LOG.Info("未填写Redis配置，跳过Redis初始化")
		return nil, nil
	}
	var client redis.UniversalClient
	// 使用集群模式
	if redisCfg.UseCluster {
		client = redis.NewClusterClient(&redis.ClusterOptions{
			Addrs:    redisCfg.ClusterAddrs,
			Password: redisCfg.Password,
		})
	} else {
		// 使用单例模式
		client = redis.NewClient(&redis.Options{
			Addr:     redisCfg.Addr,
			Password: redisCfg.Password,
			DB:       redisCfg.DB,
		})
	}
	pong, err := client.Ping(context.Background()).Result()
	if err != nil {
		global.LOG.Error("Redis连接失败，错误信息:", zap.String("name", redisCfg.Name), zap.Error(err))
		return nil, err
	}

	global.LOG.Info("Redis连接 ping 响应:", zap.String("name", redisCfg.Name), zap.String("pong", pong))
	return client, nil
}

func Redis() {
	redisClient, err := initRedisClient(global.CONFIG.Redis)
	if err != nil {
		panic(err)
	}
	global.REDIS = redisClient
}

package mongo

import (
	"context"
	"fmt"
	"sync"

	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.uber.org/zap"
)

var (
	_client  *mongo.Client
	_once    sync.Once
	_initErr error
)

func InitMongo() error {
	cfg := config.GetConfig().Mongo

	_once.Do(func() {
		client, err := mongo.Connect(context.Background(), options.Client().ApplyURI(cfg.URI))
		if err != nil {
			_initErr = fmt.Errorf("MongoDB连接失败: %w", err)
			logger.Error("MongoDB连接初始化失败", zap.Error(err))
			return
		}
		if err := client.Ping(context.Background(), nil); err != nil {
			_ = client.Disconnect(context.Background())
			_initErr = fmt.Errorf("MongoDB Ping失败: %w", err)
			logger.Error("MongoDB Ping失败", zap.Error(err))
			return
		}
		_client = client
	})

	return _initErr
}

func GetMongoClient() *mongo.Client {
	return _client
}

func GetMongoDatabase(database string) *mongo.Database {
	if _client == nil {
		return nil
	}
	if database == "" {
		database = config.GetConfig().Mongo.Database
	}
	return _client.Database(database)
}

func Close() error {
	if _client == nil {
		return nil
	}
	if err := _client.Disconnect(context.Background()); err != nil {
		logger.Error("关闭MongoDB连接失败", zap.Error(err))
		return err
	}
	logger.Info("MongoDB连接已关闭")
	return nil
}

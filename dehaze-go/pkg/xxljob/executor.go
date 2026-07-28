package xxljob

import (
	xxl "github.com/xxl-job/xxl-job-executor-go"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

var executor xxl.Executor

func Init(cfg *config.AppConfig) xxl.Executor {
	if !cfg.XxlJob.Enabled {
		logger.Info("XXL-Job 未启用，跳过执行器初始化")
		return nil
	}

	executor = xxl.NewExecutor(
		xxl.ServerAddr(cfg.XxlJob.ServerAddr),
		xxl.AccessToken(cfg.XxlJob.AccessToken),
		xxl.ExecutorPort(cfg.XxlJob.ExecutorPort),
		xxl.RegistryKey(cfg.XxlJob.RegistryKey),
	)
	executor.Init()

	logger.Info("XXL-Job 执行器初始化完成",
		zap.String("server", cfg.XxlJob.ServerAddr),
		zap.String("registryKey", cfg.XxlJob.RegistryKey),
		zap.String("port", cfg.XxlJob.ExecutorPort))

	return executor
}

func GetExecutor() xxl.Executor {
	return executor
}

func Stop() {
	if executor != nil {
		logger.Info("停止 XXL-Job 执行器")
		executor.Stop()
	}
}

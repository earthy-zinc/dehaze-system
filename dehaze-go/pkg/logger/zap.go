package logger

import (
	"fmt"
	"os"
	"sync"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
)

var (
	_globalLogger *zap.Logger
	_once         sync.Once   // 保证Init方法仅执行一次
	_defaultLog   *zap.Logger // 默认日志实例（初始化前临时使用）
)

func InitDefaultLogger() {
	cfg := zap.NewDevelopmentConfig()
	cfg.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
	defaultLog, err := cfg.Build()
	if err != nil {
		panic(fmt.Sprintf("初始化默认日志失败: %v", err))
	}
	_defaultLog = defaultLog
	_globalLogger = defaultLog
	zap.ReplaceGlobals(defaultLog)
}

func Init() error {
	var initErr error
	_once.Do(func() {
		cfg := config.GetConfig()
		if ok, err := utils.PathExists(cfg.Zap.Directory); !ok {
			if err != nil {
				initErr = fmt.Errorf("检查日志目录失败: %w", err)
				return
			}
			if err := os.Mkdir(cfg.Zap.Directory, os.ModePerm); err != nil {
				initErr = fmt.Errorf("创建日志目录失败: %w", err)
				return
			}
			zap.S().Infof("创建日志文件夹: %v", cfg.Zap.Directory)
		}

		levels := cfg.Zap.Levels()
		cores := make([]zapcore.Core, 0, len(levels))
		for i := 0; i < len(levels); i++ {
			core := NewZapCore(levels[i])
			cores = append(cores, core)
		}

		logger := zap.New(zapcore.NewTee(cores...))
		if cfg.Zap.ShowLine {
			logger = logger.WithOptions(zap.AddCaller())
		}
		zap.ReplaceGlobals(logger)
		_globalLogger = logger
	})
	return initErr
}

// Debug 调试日志（开发环境使用，生产环境可关闭）
// ctx：上下文（可传nil），msg：日志消息，fields：结构化字段（如zap.String("user_id", "123")）
func Debug(msg string, fields ...zap.Field) {
	_globalLogger.Debug(msg, fields...)
}

// Info 普通信息日志（常用，记录正常业务流程）
func Info(msg string, fields ...zap.Field) {
	_globalLogger.Info(msg, fields...)
}

// Warn 警告日志（非错误，但需关注，如缓存失效、重试）
func Warn(msg string, fields ...zap.Field) {
	_globalLogger.Warn(msg, fields...)
}

// Error 错误日志（业务错误，如数据库查询失败、接口调用失败）
func Error(msg string, fields ...zap.Field) {
	_globalLogger.Error(msg, fields...)
}

// Fatal 致命错误日志（记录后立即退出程序，如配置加载失败、数据库连接失败）
func Fatal(msg string, fields ...zap.Field) {
	_globalLogger.Fatal(msg, fields...)
}

// Panic 恐慌日志（记录后触发panic，极少使用）
func Panic(msg string, fields ...zap.Field) {
	_globalLogger.Panic(msg, fields...)
}

// Sync 刷新日志缓冲区（应用关闭时调用，保证所有日志写入文件）
func Sync() {
	_ = _globalLogger.Sync()
}

package logger

import (
	"context"
	"fmt"
	"os"
	"sync"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/trace"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
)

var (
	// 默认使用 Nop，避免在未显式 Init 时出现空指针。
	_globalLogger *zap.Logger = zap.NewNop()
	_once         sync.Once // 保证Init方法仅执行一次
)

func InitDefaultLogger() {
	cfg := zap.NewDevelopmentConfig()
	cfg.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
	defaultLog, err := cfg.Build()
	if err != nil {
		panic(fmt.Sprintf("初始化默认日志失败: %v", err))
	}
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

		// dev 环境启动时归档当天旧日志，使本次启动日志写入全新活动文件
		if cfg.Zap.ArchiveOnStartup {
			names := make([]string, 0, len(levels))
			for _, l := range levels {
				names = append(names, l.String())
			}
			archiveStartupLogs(cfg.Zap.Directory, names)
		}

		cores := make([]zapcore.Core, 0, len(levels))
		for i := 0; i < len(levels); i++ {
			// 首个档（info）挂载控制台输出，其余档（error）仅写文件，避免控制台重复
			core := NewZapCore(levels[i], levels[i].String(), i == 0)
			cores = append(cores, core)
		}

		logger := zap.New(zapcore.NewTee(cores...))
		if cfg.Zap.ShowLine {
			logger = logger.WithOptions(zap.AddCaller())
		}
		// 注入 service 字段（与 Python/Java 端对齐）
		logger = logger.With(zap.String("service", "Dehaze API"))
		zap.ReplaceGlobals(logger)
		_globalLogger = logger
	})
	return initErr
}

// WithContext 返回携带 TraceID/Method/Path/IP/UserAgent/UserID 的 logger
// 优先从 context 取缓存实例（由 Trace 中间件写入），避免重复分配
func WithContext(ctx context.Context) *zap.Logger {
	if ctx == nil {
		return _globalLogger
	}
	// 优先取缓存的 logger（整条请求链路复用同一实例）
	if l := trace.LoggerFromContext(ctx); l != nil {
		return l
	}
	// 降级：ctx 中有 TraceID 但无缓存 logger（如手动构造的 context）
	traceID := trace.FromContext(ctx)
	if traceID == "" {
		return _globalLogger
	}
	fields := []zap.Field{zap.String(trace.TraceFieldName, traceID)}
	if m := trace.MethodFromContext(ctx); m != "" {
		fields = append(fields, zap.String(trace.MethodFieldName, m))
	}
	if p := trace.PathFromContext(ctx); p != "" {
		fields = append(fields, zap.String(trace.PathFieldName, p))
	}
	if ip := trace.IPFromContext(ctx); ip != "" {
		fields = append(fields, zap.String("ip", ip))
	}
	if ua := trace.UserAgentFromContext(ctx); ua != "" {
		fields = append(fields, zap.String("user_agent", ua))
	}
	if uid := trace.UserIDFromContext(ctx); uid > 0 {
		fields = append(fields, zap.Int64("user_id", uid))
	}
	return _globalLogger.With(fields...)
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

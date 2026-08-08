package logger

import (
	"sync"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/earthyzinc/dehaze-go/pkg/config"
)

var (
	_clientLogger *zap.Logger
	_clientOnce   sync.Once
)

// GetClientLogger 返回前端日志专用 logger。
//
// 复用 Cutter 的按日期分目录 + 按大小切割 + 保留策略，写入 logs/{yyyy-MM-dd}/client.log，
// 供 filebeat 采集进入 ELK。与后端应用日志共用同一存储架构（07-日志架构设计.md §2）。
// service 固定为 client，供 logstash 按 [service]=="client" 分流到 dehaze-client-logs-* 索引。
func GetClientLogger() *zap.Logger {
	_clientOnce.Do(func() {
		cfg := config.GetConfig()
		cutter := NewCutter(
			cfg.Zap.Directory,
			"client",
			cfg.Zap.RetentionDay,
			CutterWithLayout(time.DateOnly),
			CutterWithMaxSize(cfg.Zap.MaxSize*1024*1024),
		)
		syncer := zapcore.AddSync(cutter)
		enabler := zap.LevelEnablerFunc(func(l zapcore.Level) bool {
			return l >= zapcore.InfoLevel
		})
		core := zapcore.NewCore(cfg.Zap.FileEncoder(), syncer, enabler)
		_clientLogger = zap.New(core)
		if cfg.Zap.ShowLine {
			_clientLogger = _clientLogger.WithOptions(zap.AddCaller())
		}
		_clientLogger = _clientLogger.With(zap.String("service", "client"))
	})
	return _clientLogger
}

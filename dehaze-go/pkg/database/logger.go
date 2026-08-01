package database

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
	gormLogger "gorm.io/gorm/logger"
)

// log 返回携带 trace_id 与 logger=sql 标识的 logger（从 ctx 自动提取请求上下文）
func (l *GormLogger) log(ctx context.Context) *zap.Logger {
	return logger.WithContext(ctx).Named("sql")
}

// GormLogger Gorm日志适配器
// 统一对接pkg/logger，所有数据库复用
type GormLogger struct {
	SlowThreshold         time.Duration
	LogLevel              gormLogger.LogLevel
	SkipErrRecordNotFound bool
	UseZap                bool
}

// NewGormLogger 创建Gorm日志适配器
func NewGormLogger(config *Config) gormLogger.Interface {
	return &GormLogger{
		SlowThreshold:         config.GetSlowThreshold(),
		LogLevel:              config.LogLevel(),
		SkipErrRecordNotFound: true,
		UseZap:                config.LogZap,
	}
}

// LogMode 设置日志级别
func (l *GormLogger) LogMode(level gormLogger.LogLevel) gormLogger.Interface {
	newLogger := *l
	newLogger.LogLevel = level
	return &newLogger
}

// Info 输出Info级别日志
func (l *GormLogger) Info(ctx context.Context, msg string, data ...interface{}) {
	if l.LogLevel >= gormLogger.Info {
		l.log(ctx).Info(fmt.Sprintf(msg, data...))
	}
}

// Warn 输出Warn级别日志
func (l *GormLogger) Warn(ctx context.Context, msg string, data ...interface{}) {
	if l.LogLevel >= gormLogger.Warn {
		l.log(ctx).Warn(fmt.Sprintf(msg, data...))
	}
}

// Error 输出Error级别日志
func (l *GormLogger) Error(ctx context.Context, msg string, data ...interface{}) {
	if l.LogLevel >= gormLogger.Error {
		l.log(ctx).Error(fmt.Sprintf(msg, data...))
	}
}

// Trace 输出SQL审计日志
func (l *GormLogger) Trace(ctx context.Context, begin time.Time, fc func() (sql string, rowsAffected int64), err error) {
	if l.LogLevel <= gormLogger.Silent {
		return
	}

	elapsed := time.Since(begin)
	durationMs := float64(elapsed.Nanoseconds()) / 1e6
	sql, rows := fc()

	switch {
	case err != nil && l.LogLevel >= gormLogger.Error && (!errors.Is(err, gormLogger.ErrRecordNotFound) || !l.SkipErrRecordNotFound):
		l.log(ctx).Error("SQL_ERROR",
			zap.Error(err),
			zap.String("sql", sql),
			zap.Int64("rows", rows),
			zap.Float64("duration_ms", durationMs),
		)

	case elapsed > l.SlowThreshold && l.SlowThreshold != 0 && l.LogLevel >= gormLogger.Warn:
		l.log(ctx).Warn("SLOW_SQL",
			zap.Float64("duration_ms", durationMs),
			zap.Float64("threshold_ms", float64(l.SlowThreshold.Nanoseconds())/1e6),
			zap.String("sql", sql),
			zap.Int64("rows", rows),
		)

	case l.LogLevel >= gormLogger.Info:
		l.log(ctx).Info("SQL",
			zap.Float64("duration_ms", durationMs),
			zap.String("sql", sql),
			zap.Int64("rows", rows),
		)
	}
}

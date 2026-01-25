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
		if l.UseZap {
			logger.Info(fmt.Sprintf(msg, data...))
		} else {
			fmt.Printf("[INFO] "+msg+"\n", data...)
		}
	}
}

// Warn 输出Warn级别日志
func (l *GormLogger) Warn(ctx context.Context, msg string, data ...interface{}) {
	if l.LogLevel >= gormLogger.Warn {
		if l.UseZap {
			logger.Warn(fmt.Sprintf(msg, data...))
		} else {
			fmt.Printf("[WARN] "+msg+"\n", data...)
		}
	}
}

// Error 输出Error级别日志
func (l *GormLogger) Error(ctx context.Context, msg string, data ...interface{}) {
	if l.LogLevel >= gormLogger.Error {
		if l.UseZap {
			logger.Error(fmt.Sprintf(msg, data...))
		} else {
			fmt.Printf("[ERROR] "+msg+"\n", data...)
		}
	}
}

// Trace 输出SQL日志
func (l *GormLogger) Trace(ctx context.Context, begin time.Time, fc func() (sql string, rowsAffected int64), err error) {
	if l.LogLevel <= gormLogger.Silent {
		return
	}

	elapsed := time.Since(begin)
	sql, rows := fc()

	switch {
	case err != nil && l.LogLevel >= gormLogger.Error && (!errors.Is(err, gormLogger.ErrRecordNotFound) || !l.SkipErrRecordNotFound):
		// 记录错误SQL
		if l.UseZap {
			logger.Error("SQL执行失败",
				zap.Error(err),
				zap.String("sql", sql),
				zap.Int64("rows", rows),
				zap.Duration("elapsed", elapsed),
			)
		} else {
			fmt.Printf("[ERROR] SQL执行失败 | elapsed=%v | rows=%d | sql=%s | error=%v\n", elapsed, rows, sql, err)
		}

	case elapsed > l.SlowThreshold && l.SlowThreshold != 0 && l.LogLevel >= gormLogger.Warn:
		// 记录慢查询
		if l.UseZap {
			logger.Warn("慢查询",
				zap.Duration("elapsed", elapsed),
				zap.Duration("threshold", l.SlowThreshold),
				zap.String("sql", sql),
				zap.Int64("rows", rows),
			)
		} else {
			fmt.Printf("[WARN] 慢查询 | elapsed=%v | threshold=%v | rows=%d | sql=%s\n", elapsed, l.SlowThreshold, rows, sql)
		}

	case l.LogLevel >= gormLogger.Info:
		// 记录普通SQL
		if l.UseZap {
			logger.Debug("SQL执行",
				zap.Duration("elapsed", elapsed),
				zap.String("sql", sql),
				zap.Int64("rows", rows),
			)
		} else {
			fmt.Printf("[INFO] SQL执行 | elapsed=%v | rows=%d | sql=%s\n", elapsed, rows, sql)
		}
	}
}

package options

import (
	"time"

	"go.uber.org/zap/zapcore"
)

type Zap struct {
	Level         string `mapstructure:"level" json:"level" yaml:"level" validate:"required,oneof=debug info warn error dpanic panic fatal"`
	Prefix        string `mapstructure:"prefix" json:"prefix" yaml:"prefix"`
	Format        string `mapstructure:"format" json:"format" yaml:"format" validate:"oneof=json console"`
	Directory     string `mapstructure:"directory" json:"directory"  yaml:"directory" validate:"required"`
	EncodeLevel   string `mapstructure:"encode-level" json:"encode-level" yaml:"encode-level"`
	StacktraceKey string `mapstructure:"stacktrace-key" json:"stacktrace-key" yaml:"stacktrace-key"`
	ShowLine      bool   `mapstructure:"show-line" json:"show-line" yaml:"show-line"`
	LogInConsole  bool   `mapstructure:"log-in-console" json:"log-in-console" yaml:"log-in-console"`
	RetentionDay  int    `mapstructure:"retention-day" json:"retention-day" yaml:"retention-day" validate:"gte=-1"`
	// MaxSize 单个日志文件大小上限（MB），超限归档为 {级别}.{n}.log 并开新活动文件，0 表示不按大小切割
	MaxSize int64 `mapstructure:"max-size" json:"max-size" yaml:"max-size" validate:"gte=0"`
	// ArchiveOnStartup 启动时归档当天已存在的活动日志文件（dev 用，prod 关闭以保留连续日志）
	ArchiveOnStartup bool `mapstructure:"archive-on-startup" json:"archive-on-startup" yaml:"archive-on-startup"`
}

// Levels 返回文件输出的级别分档：info（>=info，含 WARN/ERROR）与 error（>=error）两档。
// 配置的 level 字段不参与文件分档，仅控制控制台最低输出级别（见 zap_core.buildCore）。
func (c *Zap) Levels() []zapcore.Level {
	return []zapcore.Level{zapcore.InfoLevel, zapcore.ErrorLevel}
}

func (c *Zap) Encoder() zapcore.Encoder {
	config := zapcore.EncoderConfig{
		TimeKey:       "time",
		NameKey:       "name",
		LevelKey:      "level",
		CallerKey:     "caller",
		MessageKey:    "message",
		StacktraceKey: c.StacktraceKey,
		LineEnding:    zapcore.DefaultLineEnding,
		EncodeTime: func(t time.Time, encoder zapcore.PrimitiveArrayEncoder) {
			encoder.AppendString(c.Prefix + t.Format("2006-01-02 15:04:05.000"))
		},
		EncodeLevel:    c.LevelEncoder(),
		EncodeCaller:   zapcore.FullCallerEncoder,
		EncodeDuration: zapcore.SecondsDurationEncoder,
	}
	if c.Format == "json" {
		return zapcore.NewJSONEncoder(config)
	}
	return zapcore.NewConsoleEncoder(config)

}

// FileEncoder 返回用于文件输出的 JSON 编码器（结构化日志，供 ELK/Loki 采集）
func (c *Zap) FileEncoder() zapcore.Encoder {
	config := zapcore.EncoderConfig{
		TimeKey:        "timestamp",
		NameKey:        "logger",
		LevelKey:       "level",
		CallerKey:      "caller",
		MessageKey:     "message",
		StacktraceKey:  c.StacktraceKey,
		LineEnding:     zapcore.DefaultLineEnding,
		EncodeTime:     zapcore.ISO8601TimeEncoder,
		EncodeLevel:    zapcore.LowercaseLevelEncoder,
		EncodeCaller:   zapcore.FullCallerEncoder,
		EncodeDuration: zapcore.SecondsDurationEncoder,
	}
	return zapcore.NewJSONEncoder(config)
}

// LevelEncoder 根据 EncodeLevel 返回 zapcore.LevelEncoder
// Author [SliverHorn](https://github.com/SliverHorn)
func (c *Zap) LevelEncoder() zapcore.LevelEncoder {
	switch {
	case c.EncodeLevel == "LowercaseLevelEncoder": // 小写编码器(默认)
		return zapcore.LowercaseLevelEncoder
	case c.EncodeLevel == "LowercaseColorLevelEncoder": // 小写编码器带颜色
		return zapcore.LowercaseColorLevelEncoder
	case c.EncodeLevel == "CapitalLevelEncoder": // 大写编码器
		return zapcore.CapitalLevelEncoder
	case c.EncodeLevel == "CapitalColorLevelEncoder": // 大写编码器带颜色
		return zapcore.CapitalColorLevelEncoder
	default:
		return zapcore.LowercaseLevelEncoder
	}
}

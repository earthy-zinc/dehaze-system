package logger

import (
	"os"
	"sync"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/config"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

type ZapCore struct {
	level         zapcore.Level
	fileName      string
	enableConsole bool
	syncerCache   map[string]zapcore.WriteSyncer
	cacheMutex    sync.RWMutex
	zapcore.Core
}

func NewZapCore(level zapcore.Level, fileName string, enableConsole bool) *ZapCore {
	entity := &ZapCore{
		level:         level,
		fileName:      fileName,
		enableConsole: enableConsole,
		syncerCache:   make(map[string]zapcore.WriteSyncer),
	}
	entity.Core = entity.buildCore()
	return entity
}

// buildCore 构建日志 Core：文件输出使用 JSON 编码器（结构化，供 ELK/Loki 采集），
// 控制台输出使用人类可读编码器。控制台仅在首个（info）档挂载，避免重复输出。
func (z *ZapCore) buildCore(formats ...string) zapcore.Core {
	cfg := config.GetConfig()
	fileEnabler := zap.LevelEnablerFunc(func(l zapcore.Level) bool {
		return l >= z.level
	})

	fileSyncer := z.fileSyncer(formats...)
	fileCore := zapcore.NewCore(cfg.Zap.FileEncoder(), fileSyncer, fileEnabler)

	if z.enableConsole && cfg.Zap.LogInConsole {
		consoleLevel, err := zapcore.ParseLevel(cfg.Zap.Level)
		if err != nil || consoleLevel < zapcore.DebugLevel {
			consoleLevel = zapcore.InfoLevel
		}
		consoleEnabler := zap.LevelEnablerFunc(func(l zapcore.Level) bool {
			return l >= consoleLevel
		})
		consoleCore := zapcore.NewCore(cfg.Zap.Encoder(), zapcore.AddSync(os.Stdout), consoleEnabler)
		return zapcore.NewTee(fileCore, consoleCore)
	}
	return fileCore
}

// fileSyncer 返回文件切割 syncer（按级别/日期/自定义 formats 分文件），带缓存
func (z *ZapCore) fileSyncer(formats ...string) zapcore.WriteSyncer {
	cacheKey := z.fileName
	for _, f := range formats {
		cacheKey += "_" + f
	}

	z.cacheMutex.RLock()
	if syncer, ok := z.syncerCache[cacheKey]; ok {
		z.cacheMutex.RUnlock()
		return syncer
	}
	z.cacheMutex.RUnlock()

	z.cacheMutex.Lock()
	defer z.cacheMutex.Unlock()

	if syncer, ok := z.syncerCache[cacheKey]; ok {
		return syncer
	}

	cfg := config.GetConfig()
	cutter := NewCutter(
		cfg.Zap.Directory,
		z.fileName,
		cfg.Zap.RetentionDay,
		CutterWithLayout(time.DateOnly),
		CutterWithFormats(formats...),
	)

	syncer := zapcore.AddSync(cutter)
	z.syncerCache[cacheKey] = syncer
	return syncer
}

func (z *ZapCore) Enabled(level zapcore.Level) bool {
	return z.level == level
}

func (z *ZapCore) With(fields []zapcore.Field) zapcore.Core {
	return z.Core.With(fields)
}

func (z *ZapCore) Check(entry zapcore.Entry, check *zapcore.CheckedEntry) *zapcore.CheckedEntry {
	if z.Enabled(entry.Level) {
		return check.AddCore(entry, z)
	}
	return check
}

func (z *ZapCore) Write(entry zapcore.Entry, fields []zapcore.Field) error {
	for i := 0; i < len(fields); i++ {
		if fields[i].Key == "business" || fields[i].Key == "folder" || fields[i].Key == "directory" {
			core := z.buildCore(fields[i].String)
			return core.Write(entry, fields)
		}
	}
	return z.Core.Write(entry, fields)
}

func (z *ZapCore) Sync() error {
	return z.Core.Sync()
}

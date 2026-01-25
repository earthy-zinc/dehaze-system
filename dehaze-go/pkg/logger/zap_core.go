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
	level       zapcore.Level
	syncerCache map[string]zapcore.WriteSyncer
	cacheMutex  sync.RWMutex
	zapcore.Core
}

func NewZapCore(level zapcore.Level) *ZapCore {
	entity := &ZapCore{
		level:       level,
		syncerCache: make(map[string]zapcore.WriteSyncer),
	}
	syncer := entity.WriteSyncer()
	levelEnabler := zap.LevelEnablerFunc(func(l zapcore.Level) bool {
		return l == level
	})
	cfg := config.GetConfig()
	entity.Core = zapcore.NewCore(cfg.Zap.Encoder(), syncer, levelEnabler)
	return entity
}

func (z *ZapCore) WriteSyncer(formats ...string) zapcore.WriteSyncer {
	cacheKey := z.level.String()
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
		z.level.String(),
		cfg.Zap.RetentionDay,
		CutterWithLayout(time.DateOnly),
		CutterWithFormats(formats...),
	)

	var syncer zapcore.WriteSyncer
	if cfg.Zap.LogInConsole {
		syncer = zapcore.AddSync(zapcore.NewMultiWriteSyncer(os.Stdout, cutter))
	} else {
		syncer = zapcore.AddSync(cutter)
	}

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
	cfg := config.GetConfig()
	for i := 0; i < len(fields); i++ {
		if fields[i].Key == "business" || fields[i].Key == "folder" || fields[i].Key == "directory" {
			syncer := z.WriteSyncer(fields[i].String)
			core := zapcore.NewCore(cfg.Zap.Encoder(), syncer, z.level)
			return core.Write(entry, fields)
		}
	}
	return z.Core.Write(entry, fields)
}

func (z *ZapCore) Sync() error {
	return z.Core.Sync()
}

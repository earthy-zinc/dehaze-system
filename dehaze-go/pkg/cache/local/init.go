package local

import (
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/songzhibin97/gkit/cache/local_cache"
	"github.com/songzhibin97/gkit/options"
)

func InitLocalCache() *LocalCache {
	cfg := config.GetConfig().Cache.Local

	var opts []options.Option
	opts = append(opts, local_cache.SetDefaultExpire(time.Duration(cfg.DefaultExpire)*time.Second))
	c := local_cache.NewCache(opts...)

	return NewLocalCache(c)
}

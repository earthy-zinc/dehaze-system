package initialize

import (
	"time"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/songzhibin97/gkit/cache/local_cache"
)

func LocalCache() {
	ttl := time.Duration(global.CONFIG.JWT.TTL)
	global.BlackCache = local_cache.NewCache(
		local_cache.SetDefaultExpire(ttl),
	)
}

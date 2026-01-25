package protection

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
)

const (
	// NullValueMarker 空值标记
	NullValueMarker = "__NULL__"
)

// NullCache 空值缓存实现
type NullCache struct {
	cache      types.ICache
	expiration time.Duration
}

// NewNullCache 创建空值缓存
func NewNullCache(cache types.ICache, expireSeconds int) *NullCache {
	if expireSeconds <= 0 {
		expireSeconds = 60 // 默认1分钟
	}
	return &NullCache{
		cache:      cache,
		expiration: time.Duration(expireSeconds) * time.Second,
	}
}

// IsNullValue 检查值是否为空值标记
func (nc *NullCache) IsNullValue(value string) bool {
	return value == NullValueMarker
}

// SetNull 设置空值缓存
func (nc *NullCache) SetNull(ctx context.Context, key string, expiration time.Duration) error {
	if expiration == 0 {
		expiration = nc.expiration
	}
	return nc.cache.Set(ctx, key, NullValueMarker, expiration)
}

// GetNullValue 获取空值标记字符串
func (nc *NullCache) GetNullValue() string {
	return NullValueMarker
}

package error

import "errors"

var (
	ErrKeyNotFound         = errors.New("缓存键不存在")
	ErrCacheNotInitialized = errors.New("缓存未初始化")
	ErrBackendUnavailable  = errors.New("缓存后端不可用")
	ErrInvalidKey          = errors.New("无效的缓存键")
	ErrInvalidExpiration   = errors.New("无效的过期时间")
	ErrOperationFailed     = errors.New("缓存操作失败")
	ErrFallbackFailed      = errors.New("降级操作失败")
)

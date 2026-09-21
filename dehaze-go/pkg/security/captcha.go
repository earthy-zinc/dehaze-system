package security

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/cache/errs"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/mojocn/base64Captcha"
	"go.uber.org/zap"
)

var (
	cacheClient    types.ICache
	singletonStore base64Captcha.Store
	once           sync.Once
)

func GetCaptchaStore() base64Captcha.Store {
	once.Do(func() {
		cacheClient = cache.GetCache()
		singletonStore = NewCacheStore()
	})
	return singletonStore
}

func NewCacheStore() *CacheStore {
	return &CacheStore{
		Expiration: time.Second * 120,
		PreKey:     common.CaptchaCodePrefix,
		Context:    context.Background(),
	}
}

type CacheStore struct {
	Expiration time.Duration
	PreKey     string
	Context    context.Context
}

func (rs *CacheStore) Set(id string, value string) error {
	err := cacheClient.Set(rs.Context, rs.PreKey+id, value, rs.Expiration)
	if err != nil {
		logger.Error("RedisStoreSetError!", zap.Error(err))
		return err
	}
	return nil
}

func (rs *CacheStore) Get(key string, clear bool) string {
	// clear=true 是"取走即消费"：必须用 GetDel 原子完成，否则并发提交同一 captchaKey 时
	// 两个请求都能读到值（并发登录双双成功）。python `verify_captcha_status` 用 Redis GETDEL 同口径。
	if clear {
		val, err := cacheClient.GetDel(rs.Context, rs.PreKey+key)
		if err != nil {
			// 未命中（已被并发消费或已过期）属正常路径，不记错误日志
			if !errors.Is(err, errs.ErrKeyNotFound) {
				logger.Error("RedisStoreGetDelError!", zap.Error(err))
			}
			return ""
		}
		return val
	}

	val, err := cacheClient.Get(rs.Context, rs.PreKey+key)
	if err != nil {
		logger.Error("RedisStoreGetError!", zap.Error(err))
		return ""
	}
	return val
}

func (rs *CacheStore) Verify(id, answer string, clear bool) bool {
	v := rs.Get(id, clear)
	return v == answer
}

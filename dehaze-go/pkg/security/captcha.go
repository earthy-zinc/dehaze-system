package security

import (
	"context"
	"sync"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/cache"
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
		Expiration: time.Second * 180,
		PreKey:     common.CaptchaCodePrefix,
		Context:    context.TODO(),
	}
}

type CacheStore struct {
	Expiration time.Duration
	PreKey     string
	Context    context.Context
}

func (rs *CacheStore) UseWithCtx(ctx context.Context) *CacheStore {
	if ctx == nil {
		rs.Context = ctx
	}
	return rs
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
	val, err := cacheClient.Get(rs.Context, key)
	if err != nil {
		logger.Error("RedisStoreGetError!", zap.Error(err))
		return ""
	}
	if clear {
		err := cacheClient.Delete(rs.Context, key)
		if err != nil {
			logger.Error("RedisStoreClearError!", zap.Error(err))
			return ""
		}
	}
	return val
}

func (rs *CacheStore) Verify(id, answer string, clear bool) bool {
	key := rs.PreKey + id
	v := rs.Get(key, clear)
	return v == answer
}

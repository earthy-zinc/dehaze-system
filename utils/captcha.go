package utils

import (
	"context"
	"sync"
	"time"

	"github.com/earthyzinc/dehaze-go/common"
	"github.com/earthyzinc/dehaze-go/global"
	"github.com/mojocn/base64Captcha"
	"go.uber.org/zap"
)

var (
	// 单例 Store 实例
	singletonStore base64Captcha.Store
	// 保证初始化只执行一次
	once sync.Once
)

func GetCaptchaStore() base64Captcha.Store {
	once.Do(func() {
		if global.REDIS != nil {
			singletonStore = NewDefaultRedisStore()
		} else {
			singletonStore = base64Captcha.DefaultMemStore
		}
	})
	return singletonStore
}

func NewDefaultRedisStore() *RedisStore {
	return &RedisStore{
		Expiration: time.Second * 180,
		PreKey:     common.CAPTCHA_CODE_PREFIX,
		Context:    context.TODO(),
	}
}

type RedisStore struct {
	Expiration time.Duration
	PreKey     string
	Context    context.Context
}

func (rs *RedisStore) UseWithCtx(ctx context.Context) *RedisStore {
	if ctx == nil {
		rs.Context = ctx
	}
	return rs
}

func (rs *RedisStore) Set(id string, value string) error {
	err := global.REDIS.Set(rs.Context, rs.PreKey+id, value, rs.Expiration).Err()
	if err != nil {
		global.LOG.Error("RedisStoreSetError!", zap.Error(err))
		return err
	}
	return nil
}

func (rs *RedisStore) Get(key string, clear bool) string {
	val, err := global.REDIS.Get(rs.Context, key).Result()
	if err != nil {
		global.LOG.Error("RedisStoreGetError!", zap.Error(err))
		return ""
	}
	if clear {
		err := global.REDIS.Del(rs.Context, key).Err()
		if err != nil {
			global.LOG.Error("RedisStoreClearError!", zap.Error(err))
			return ""
		}
	}
	return val
}

func (rs *RedisStore) Verify(id, answer string, clear bool) bool {
	key := rs.PreKey + id
	v := rs.Get(key, clear)
	return v == answer
}

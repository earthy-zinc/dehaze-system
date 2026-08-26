//go:build integration

package auth_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	redisClient "github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"golang.org/x/crypto/bcrypt"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/service/auth"
	"github.com/earthyzinc/dehaze-go/internal/service/mocks"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/cache/errs"
	cacheredis "github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
)

// 为什么不用 TestMain：本目录既有 package auth 的 TestMain（auth_service_test.go），而 Go 把
// 内部测试包与外部测试包（auth_test）链接进同一个测试二进制，TestMain 全局唯一，同二进制下
// 第二个 TestMain 必然链接冲突（"multiple definitions of TestMain"），且既有文件禁止改动。
// 因此由各用例首行幂等初始化（LoadTestConfig/cache.Init 内部均 sync.Once，gin.SetMode 幂等），
// 效果等同 TestMain 的进程级一次性环境构造。
//
// 为什么自建 Redis client 而不用全局 cache.GetCache()/cacheredis.GetClient()：内部测试包的
// miniredis 用例（auth_service_miniredis_test.go）在同一测试二进制中抢先触发 pkg/cache 与
// pkg/cache/redis 的进程级 sync.Once，全局缓存被绑定到 miniredis（DB=0），无法重置。集成
// 用例必须触达真实 Redis（db=4），故基于 config.test.yaml 显式构造独立 client。

// realRedisClient 基于 config.test.yaml 构造并返回真实 Redis（db=4）client，测试结束自动关闭。
func realRedisClient(t *testing.T) *redisClient.Client {
	t.Helper()
	cfg := testutil.LoadTestConfig(t) // 加载 config.test.yaml（.env 展开），全局 config.Config 的 Redis db=4
	client := redisClient.NewClient(&redisClient.Options{
		Addr:     cfg.Cache.Redis.Addr,
		Password: cfg.Cache.Redis.Password,
		DB:       cfg.Cache.Redis.DB,
	})
	t.Cleanup(func() { _ = client.Close() })
	return client
}

// newRealCache 基于真实 Redis（db=4）构造 ICache，供缓存语义用例与 AuthService 注入。
func newRealCache(t *testing.T) types.ICache {
	t.Helper()
	gin.SetMode(gin.TestMode)
	return cacheredis.NewRedisCache(realRedisClient(t))
}

// cleanupRedis 按前缀删除真实 Redis（db=4）测试键（含显式 extra 键）。禁止 FLUSHDB：db=4 可能与他人共享。
// 注意：此函数在 t.Cleanup 中执行，不能再注册 Cleanup，故自建 client 并 defer Close。
func cleanupRedis(t *testing.T, prefix string, extra ...string) {
	t.Helper()
	cfg := testutil.LoadTestConfig(t)
	client := redisClient.NewClient(&redisClient.Options{
		Addr:     cfg.Cache.Redis.Addr,
		Password: cfg.Cache.Redis.Password,
		DB:       cfg.Cache.Redis.DB,
	})
	defer client.Close()
	ctx := context.Background()
	keys := append([]string{}, extra...)
	if prefix != "" {
		if matched, err := client.Keys(ctx, prefix+"*").Result(); err == nil {
			keys = append(keys, matched...)
		}
	}
	if len(keys) > 0 {
		_ = client.Del(ctx, keys...).Err()
	}
}

// setCaptcha 向全局验证码 store 写入验证码，供 Login 内 VerifyCaptcha 命中。
// 验证码 store 绑定全局 cache.GetCache()：若内部测试的 miniredis 用例已抢先初始化则绑定
// miniredis，验证码读写同一 store，与集成用例触达真实 Redis 的断言互不影响；否则按
// config.test.yaml（db=4）初始化全局缓存后再绑定。
func setCaptcha(t *testing.T, key, code string) {
	t.Helper()
	testutil.LoadTestConfig(t)
	if cache.GetCache() == nil {
		if _, err := cache.Init(); err != nil {
			t.Fatalf("初始化全局缓存失败: %v", err)
		}
	}
	if err := security.GetCaptchaStore().Set(key, code); err != nil {
		t.Fatalf("写入验证码失败: %v", err)
	}
}

func TestIntegration_CacheSetGetTTL(t *testing.T) {
	c := newRealCache(t)
	ctx := context.Background()
	key := t.Name() + ":kv"
	t.Cleanup(func() { cleanupRedis(t, t.Name()) })

	if err := c.Set(ctx, key, "hello-integration", 300*time.Second); err != nil {
		t.Fatalf("Set 失败: %v", err)
	}
	val, err := c.Get(ctx, key)
	assert.NoError(t, err)
	assert.Equal(t, "hello-integration", val)

	ttl, err := c.TTL(ctx, key)
	assert.NoError(t, err)
	assert.GreaterOrEqual(t, ttl, 250*time.Second)
	assert.LessOrEqual(t, ttl, 300*time.Second)

	// 未命中键应返回 ErrKeyNotFound，验证真实缓存缺键语义
	_, err = c.Get(ctx, t.Name()+":missing")
	assert.ErrorIs(t, err, errs.ErrKeyNotFound)
}

func TestIntegration_CacheIncrAtomic(t *testing.T) {
	c := newRealCache(t)
	ctx := context.Background()
	key := t.Name() + ":counter"
	t.Cleanup(func() { cleanupRedis(t, t.Name()) })

	for i := 1; i <= 3; i++ {
		n, err := c.Incr(ctx, key)
		assert.NoError(t, err)
		assert.Equal(t, int64(i), n, "第 %d 次 INCR 应返回 %d", i, i)
	}
	val, err := c.Get(ctx, key)
	assert.NoError(t, err)
	assert.Equal(t, "3", val)
}

func TestIntegration_RedisDB4Isolation(t *testing.T) {
	cfg := testutil.LoadTestConfig(t)
	ctx := context.Background()
	key := t.Name() + ":probe"
	t.Cleanup(func() { cleanupRedis(t, t.Name()) })

	assert.Equal(t, 4, cfg.Cache.Redis.DB, "config.test.yaml 应指向 Redis db=4")

	// 直连 db=4 写入探针键，再以相同 addr/password 直连 db=0 交叉验证隔离
	db4 := redisClient.NewClient(&redisClient.Options{
		Addr:     cfg.Cache.Redis.Addr,
		Password: cfg.Cache.Redis.Password,
		DB:       cfg.Cache.Redis.DB,
	})
	defer db4.Close()
	if err := db4.Set(ctx, key, "db4-only", time.Minute).Err(); err != nil {
		t.Fatalf("Set 失败: %v", err)
	}

	db0 := redisClient.NewClient(&redisClient.Options{
		Addr:     cfg.Cache.Redis.Addr,
		Password: cfg.Cache.Redis.Password,
		DB:       0,
	})
	defer db0.Close()
	_, err := db0.Get(ctx, key).Result()
	assert.ErrorIs(t, err, redisClient.Nil, "db=0 不应看到 db=4 的键")
}

func TestIntegration_LoginSuccessWritesSession(t *testing.T) {
	c := newRealCache(t)
	ctx := context.Background()
	username := strings.ToLower(t.Name())
	clientIP := t.Name() + "-ip"
	captchaKey := t.Name() + "-captcha"
	captchaCode := "8842"

	// 模拟真实 userService 语义：用户表中存的是 bcrypt 哈希而非明文
	hash, err := bcrypt.GenerateFromPassword([]byte("Secret@123"), bcrypt.DefaultCost)
	if err != nil {
		t.Fatalf("生成密码哈希失败: %v", err)
	}
	user := &model.UserAuthInfo{
		UserId:    1001,
		Username:  username,
		Nickname:  "张三",
		DeptId:    10,
		Status:    1,
		Roles:     []string{"admin"},
		Perms:     []string{"sys:user:list"},
		DataScope: 1,
		Password:  string(hash),
	}

	mockUser := mocks.NewMockIUserService(t)
	mockUser.EXPECT().Login(ctx, mock.MatchedBy(func(u *model.SysUser) bool {
		return u.Username == username && u.Password == "Secret@123"
	})).Return(user, nil).Once()

	setCaptcha(t, captchaKey, captchaCode)

	svc := auth.NewAuthService(c, mockUser, nil, nil)
	result, err := svc.Login(ctx, &bo.LoginRequest{
		Username:    username,
		Password:    "Secret@123",
		CaptchaKey:  captchaKey,
		CaptchaCode: captchaCode,
	}, clientIP, "Mozilla/5.0 (Windows NT 10.0) Chrome/120")

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.NotEmpty(t, result.SessionID)

	sessionKey := common.SessionPrefix + result.SessionID
	t.Cleanup(func() { cleanupRedis(t, t.Name(), sessionKey) })

	// session 真实写入 db=4，TTL≈SessionTTL（7 天）
	val, err := c.Get(ctx, sessionKey)
	assert.NoError(t, err)
	assert.Contains(t, val, `"userId":1001`)
	assert.Contains(t, val, `"username":"`+username+`"`)
	assert.Contains(t, val, "ROLE_admin")

	ttl, err := c.TTL(ctx, sessionKey)
	assert.NoError(t, err)
	assert.Greater(t, ttl, 160*time.Hour)
	assert.LessOrEqual(t, ttl, middleware.SessionTTL)

	// 登录成功后失败计数被真实重置（无残留）
	_, err = c.Get(ctx, "login:fail:ip:"+clientIP)
	assert.ErrorIs(t, err, errs.ErrKeyNotFound)
}

func TestIntegration_LoginFailureIncrementsFailCount(t *testing.T) {
	c := newRealCache(t)
	ctx := context.Background()
	username := strings.ToLower(t.Name())
	clientIP := t.Name() + "-ip"
	captchaKey := t.Name() + "-captcha"
	captchaCode := "1234"
	ipKey := "login:fail:ip:" + clientIP
	userKey := "login:fail:" + username
	t.Cleanup(func() { cleanupRedis(t, t.Name(), ipKey, userKey) })

	mockUser := mocks.NewMockIUserService(t)
	mockUser.EXPECT().Login(ctx, mock.Anything).
		Return(nil, common.NewBizError(common.USER_NOT_EXIST, "用户不存在")).Once()

	setCaptcha(t, captchaKey, captchaCode)

	svc := auth.NewAuthService(c, mockUser, nil, nil)
	result, err := svc.Login(ctx, &bo.LoginRequest{
		Username:    username,
		Password:    "WrongPass@1",
		CaptchaKey:  captchaKey,
		CaptchaCode: captchaCode,
	}, clientIP, "")

	assert.Nil(t, result)
	assert.Error(t, err)

	// fail 计数键真实写入 db=4 且 TTL≈LoginFailLockTime（300s）
	for key, want := range map[string]string{ipKey: "1", userKey: "1"} {
		val, getErr := c.Get(ctx, key)
		assert.NoError(t, getErr, "键 %s 应存在", key)
		assert.Equal(t, want, val, "键 %s 计数不符", key)
		ttl, ttlErr := c.TTL(ctx, key)
		assert.NoError(t, ttlErr, "键 %s 应有 TTL", key)
		assert.GreaterOrEqual(t, ttl, 250*time.Second)
		assert.LessOrEqual(t, ttl, 300*time.Second)
	}
}

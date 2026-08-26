package auth

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	redisClient "github.com/redis/go-redis/v9"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/service/mocks"
	"github.com/earthyzinc/dehaze-go/pkg/cache"
	cacheredis "github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// 进程级 miniredis 实例：既作为验证码 store 的全局后端（security.GetCaptchaStore 绑定全局 cache），
// 也作为注入 AuthService 的真实 ICache 后端。单测不触达真实 Redis(127.0.0.1:6379)。
var (
	captchaBackendOnce sync.Once
	captchaMr          *miniredis.Miniredis
)

func initCaptchaBackend(t *testing.T) *miniredis.Miniredis {
	t.Helper()
	captchaBackendOnce.Do(func() {
		mr, err := miniredis.Run()
		if err != nil {
			t.Fatalf("启动 miniredis 失败: %v", err)
		}
		captchaMr = mr
		// 把全局缓存指向 miniredis，使 security.GetCaptchaStore 首次绑定可用后端
		config.Config.Cache = options.Cache{
			Type:  "redis",
			Redis: options.Redis{Enabled: true, Addr: mr.Addr(), DB: 0},
		}
		if _, err := cache.Init(); err != nil {
			t.Fatalf("初始化全局缓存失败: %v", err)
		}
		// 触发验证码 store 首次绑定（绑定到上面的 miniredis 全局 cache）
		_ = security.GetCaptchaStore()
	})
	return captchaMr
}

// newRealCacheService 用真实 miniredis 缓存构造 AuthService，返回实例与底层 miniredis 供断言真实语义。
// memberService 传 nil：当前 internal/service/mocks 未生成 IMemberService（属批次2职责），
// Register 成功路径需调用 InitDefaultMember，故本文件仅覆盖不触发 memberService 的 Register 分支。
func newRealCacheService(t *testing.T, userService *mocks.MockIUserService) (*AuthService, *miniredis.Miniredis) {
	t.Helper()
	mr := initCaptchaBackend(t)
	// 进程级 miniredis 实例在所有用例间共享，构造前清空避免跨用例状态污染
	mr.FlushAll()
	// 注入实例连同一 miniredis 实例，独立 client 不影响全局验证码 store
	realCache := cacheredis.NewRedisCache(redisClient.NewClient(&redisClient.Options{Addr: mr.Addr(), DB: 0}))
	svc := NewAuthService(realCache, userService, nil, nil).(*AuthService)
	return svc, mr
}

// setCaptcha 写入一个能通过校验的验证码（走全局 miniredis 验证码 store）。
func setCaptcha(t *testing.T, key, code string) {
	t.Helper()
	initCaptchaBackend(t)
	if err := security.GetCaptchaStore().Set(key, code); err != nil {
		t.Fatalf("写入验证码失败: %v", err)
	}
}

func TestLogin_Success_WritesSession(t *testing.T) {
	mockUser := mocks.NewMockIUserService(t)
	svc, mr := newRealCacheService(t, mockUser)

	ctx := context.Background()
	clientIP := "10.0.0.5"
	captchaKey := "ck-login-success"
	captchaCode := "8842"
	setCaptcha(t, captchaKey, captchaCode)

	user := newTestUser()
	mockUser.EXPECT().Login(ctx, mock.MatchedBy(func(u *model.SysUser) bool {
		return u.Username == "zhangsan" && u.Password == "Secret@123"
	})).Return(user, nil).Once()

	req := &bo.LoginRequest{
		Username:    "zhangsan",
		Password:    "Secret@123",
		CaptchaKey:  captchaKey,
		CaptchaCode: captchaCode,
	}
	result, err := svc.Login(ctx, req, clientIP, "Mozilla/5.0 (Windows NT 10.0) Chrome/120 Safari/537")

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.NotEmpty(t, result.SessionID)
	assert.Equal(t, int64(1001), result.User.ID)
	assert.Equal(t, "zhangsan", result.User.Username)
	assert.Equal(t, "张三", result.User.Nickname)

	// 断言真实缓存语义：session 已写入且 TTL 约为 7 天（SessionTTL）
	sessionKey := common.SessionPrefix + result.SessionID
	val, getErr := mr.Get(sessionKey)
	assert.NoError(t, getErr)
	assert.Contains(t, val, "\"userId\":1001")
	assert.Contains(t, val, "\"username\":\"zhangsan\"")
	assert.Contains(t, val, "ROLE_admin")
	ttl := mr.TTL(sessionKey)
	assert.Greater(t, ttl, 6*time.Hour)
	assert.LessOrEqual(t, ttl, 7*24*time.Hour)
	// 登录成功后失败计数应被重置（无残留）
	_, ipErr := mr.Get("login:fail:ip:" + clientIP)
	assert.Error(t, ipErr)
}

func TestLogin_PasswordWrong_WritesFailCount(t *testing.T) {
	mockUser := mocks.NewMockIUserService(t)
	svc, mr := newRealCacheService(t, mockUser)

	ctx := context.Background()
	clientIP := "10.0.0.9"
	captchaKey := "ck-pwd-wrong"
	captchaCode := "1234"
	setCaptcha(t, captchaKey, captchaCode)

	mockUser.EXPECT().Login(ctx, mock.Anything).Return(nil, common.NewBizError(common.USER_NOT_EXIST, "用户不存在")).Once()

	req := &bo.LoginRequest{
		Username:    "zhangsan",
		Password:    "WrongPass@1",
		CaptchaKey:  captchaKey,
		CaptchaCode: captchaCode,
	}
	result, err := svc.Login(ctx, req, clientIP, "")

	assert.Nil(t, result)
	assertBizError(t, err, common.USER_NOT_EXIST)

	// 断言真实缓存语义：IP 失败计数键写入值=1，TTL≈300s（LoginFailLockTime）
	ipKey := "login:fail:ip:" + clientIP
	ipVal, getErr := mr.Get(ipKey)
	assert.NoError(t, getErr)
	assert.Equal(t, "1", ipVal)
	ipTTL := mr.TTL(ipKey)
	assert.Greater(t, ipTTL, 4*time.Minute)
	assert.LessOrEqual(t, ipTTL, 5*time.Minute)

	// 用户名失败计数键同样写入
	userKey := "login:fail:zhangsan"
	userVal, getErr := mr.Get(userKey)
	assert.NoError(t, getErr)
	assert.Equal(t, "1", userVal)
}

func TestLogin_UserDisabled(t *testing.T) {
	mockUser := mocks.NewMockIUserService(t)
	svc, mr := newRealCacheService(t, mockUser)

	ctx := context.Background()
	clientIP := "10.0.0.11"
	captchaKey := "ck-disabled"
	captchaCode := "5678"
	setCaptcha(t, captchaKey, captchaCode)

	disabled := newTestUser()
	disabled.Status = 0
	mockUser.EXPECT().Login(ctx, mock.Anything).Return(disabled, nil).Once()

	req := &bo.LoginRequest{
		Username:    "zhangsan",
		Password:    "Secret@123",
		CaptchaKey:  captchaKey,
		CaptchaCode: captchaCode,
	}
	result, err := svc.Login(ctx, req, clientIP, "")

	assert.Nil(t, result)
	assertBizError(t, err, common.USER_ACCOUNT_LOCKED)
	// 被禁用用户不应写入 session
	var sessionKeys []string
	for _, k := range mr.Keys() {
		if strings.HasPrefix(k, common.SessionPrefix) {
			sessionKeys = append(sessionKeys, k)
		}
	}
	assert.Empty(t, sessionKeys, "被禁用用户不应写入任何 session")
}

func TestLogin_CaptchaError(t *testing.T) {
	mockUser := mocks.NewMockIUserService(t)
	svc, mr := newRealCacheService(t, mockUser)

	ctx := context.Background()
	clientIP := "10.0.0.13"
	captchaKey := "ck-captcha-err"
	captchaCode := "0000"
	// 写入一个不同的码，使 VerifyCaptcha 不匹配
	setCaptcha(t, captchaKey, "9999")

	req := &bo.LoginRequest{
		Username:    "zhangsan",
		Password:    "Secret@123",
		CaptchaKey:  captchaKey,
		CaptchaCode: captchaCode,
	}
	result, err := svc.Login(ctx, req, clientIP, "")

	assert.Nil(t, result)
	assertBizError(t, err, common.VERIFY_CODE_ERROR)
	// 验证码错误也会递增失败计数
	ipVal, getErr := mr.Get("login:fail:ip:" + clientIP)
	assert.NoError(t, getErr)
	assert.Equal(t, "1", ipVal)
}

func TestRegister_CaptchaError(t *testing.T) {
	mockUser := mocks.NewMockIUserService(t)
	svc, _ := newRealCacheService(t, mockUser)

	ctx := context.Background()
	clientIP := "10.0.0.25"
	captchaKey := "ck-reg-captcha-err"
	setCaptcha(t, captchaKey, "1111")

	req := &bo.RegisterRequest{
		Username:    "abc",
		Nickname:    "abc",
		Password:    "Reg@2024",
		CaptchaKey:  captchaKey,
		CaptchaCode: "2222", // 不匹配
	}
	result, err := svc.Register(ctx, req, clientIP)

	assert.Nil(t, result)
	assertBizError(t, err, common.VERIFY_CODE_ERROR)
	mockUser.AssertNotCalled(t, "Register", mock.Anything, mock.Anything, mock.Anything, mock.Anything)
}

func TestRegister_UserServiceError(t *testing.T) {
	mockUser := mocks.NewMockIUserService(t)
	svc, _ := newRealCacheService(t, mockUser)

	ctx := context.Background()
	clientIP := "10.0.0.23"
	captchaKey := "ck-reg-err"
	captchaCode := "8765"
	setCaptcha(t, captchaKey, captchaCode)

	mockUser.EXPECT().Register(ctx, "dupuser", "dup", "Reg@2024").
		Return(nil, int8(0), common.NewBizError(common.USER_NOT_EXIST, "用户名已存在")).Once()

	req := &bo.RegisterRequest{
		Username:    "dupuser",
		Nickname:    "dup",
		Password:    "Reg@2024",
		CaptchaKey:  captchaKey,
		CaptchaCode: captchaCode,
	}
	result, err := svc.Register(ctx, req, clientIP)

	assert.Nil(t, result)
	assertBizError(t, err, common.USER_NOT_EXIST)
}

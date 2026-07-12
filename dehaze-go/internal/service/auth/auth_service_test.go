package auth

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/service/mocks"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// ============================================================
// TestMain: 预设最小化全局配置，使 security 包函数可正常工作
// ============================================================

// 测试专用常量，与全局配置完全解耦
const (
	testJWTKey     = "test-jwt-secret-key-for-unit-test"
	testJWTTTL     = 3600 * time.Second
	testRefreshTTL = 86400 * time.Second
)

func TestMain(m *testing.M) {
	// auth_service 内部调用 security.NewJWT() 仍需全局 JWT 配置
	// 测试辅助函数通过 NewJWTWithConfig 注入，两者使用相同的 key 和 TTL 以保证一致性
	config.Config = &config.AppConfig{
		JWT: options.JWT{
			Key:             testJWTKey,
			TTL:             int64(testJWTTTL.Seconds()),
			RefreshTokenTTL: int64(testRefreshTTL.Seconds()),
		},
		System: options.System{
			UseMultiPoint:     false,
			LoginFailLimit:    5,
			LoginFailLockTime: 300,
		},
	}
	os.Exit(m.Run())
}

// ============================================================
// 测试辅助函数
// ============================================================

// testFixture 聚合测试所需的 mock 对象和被测服务
type testFixture struct {
	cache       *mocks.MockICache
	userService *mocks.MockIUserService
	authService *AuthService
}

// setupTest 创建测试夹具，返回 mock 对象和被测服务实例
func setupTest(t *testing.T) *testFixture {
	t.Helper()
	mockCache := mocks.NewMockICache(t)
	mockUserService := mocks.NewMockIUserService(t)
	svc := NewAuthService(mockCache, mockUserService).(*AuthService)
	return &testFixture{
		cache:       mockCache,
		userService: mockUserService,
		authService: svc,
	}
}

// assertBizError 断言错误为 BizError 且错误码匹配
func assertBizError(t *testing.T, err error, expectedCode *common.ResultCode) {
	t.Helper()
	assert.Error(t, err)
	var bizErr *common.BizError
	assert.True(t, errors.As(err, &bizErr), "期望 BizError 类型，实际: %T", err)
	assert.Equal(t, expectedCode, bizErr.Code(), "错误码不匹配")
}

// newTestJWT 创建测试专用 JWT 实例，不依赖全局配置
func newTestJWT() *security.JWT {
	return security.NewJWTWithConfig([]byte(testJWTKey), testJWTTTL)
}

// newTestUser 创建带有业务含义的测试用户认证信息
func newTestUser() *model.UserAuthInfo {
	return &model.UserAuthInfo{
		UserId:    1001,
		Username:  "zhangsan",
		Nickname:  "张三",
		DeptId:    10,
		Status:    1,
		Roles:     []string{"admin"},
		Perms:     []string{"sys:user:list", "sys:role:list"},
		DataScope: 1,
	}
}

// generateTestToken 通过实例方法生成测试 Token，不依赖全局配置
func generateTestToken(t *testing.T, user *model.UserAuthInfo) (accessToken, refreshToken string) {
	t.Helper()
	j := newTestJWT()
	access, refresh, _, _, err := j.LoginTokenWithRefresh(user, testRefreshTTL)
	assert.NoError(t, err, "生成测试Token不应失败")
	assert.NotEmpty(t, access)
	assert.NotEmpty(t, refresh)
	return access, refresh
}

// mockCacheGetNotFound 模拟缓存未命中（key不存在）
func mockCacheGetNotFound(cache *mocks.MockICache, key string) {
	cache.EXPECT().Get(mock.Anything, key).Return("", errors.New("key not found")).Once()
}

// ============================================================
// Login 测试用例
// ============================================================

// TestLogin_Success_Skip 说明：
// Login 成功路径依赖 security.GetCaptchaStore() 全局单例（内部调用 cache.GetCache()），
// 当前架构下无法在纯单测中 mock。
// 可 mock 的分支（锁定检查、用户名标准化、失败计数）由下方测试覆盖。
// Login 完整成功路径应在集成测试中验证。

func TestLogin_NilRequest(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	result, err := f.authService.Login(ctx, nil, "192.168.1.100")

	assert.Nil(t, result)
	assertBizError(t, err, common.PARAM_ERROR)
	assert.Contains(t, err.Error(), "登录请求不能为空")
}

func TestLogin_IPLocked(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()
	clientIP := "192.168.1.100"

	req := &bo.LoginRequest{
		Username:    "zhangsan",
		Password:    "Admin@2024",
		CaptchaKey:  "key-001",
		CaptchaCode: "123456",
	}

	// IP 维度已达到失败上限（5次）
	f.cache.EXPECT().Get(mock.Anything, "login:fail:ip:"+clientIP).Return("5", nil).Once()

	result, err := f.authService.Login(ctx, req, clientIP)

	assert.Nil(t, result)
	assertBizError(t, err, common.PASSWORD_ENTER_EXCEED_LIMIT)
	assert.Contains(t, err.Error(), "IP已临时锁定")
}

func TestLogin_UsernameLocked(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()
	clientIP := "192.168.1.100"

	req := &bo.LoginRequest{
		Username:    "zhangsan",
		Password:    "Admin@2024",
		CaptchaKey:  "key-001",
		CaptchaCode: "123456",
	}

	// IP 维度未锁定
	mockCacheGetNotFound(f.cache, "login:fail:ip:"+clientIP)
	// 用户名维度已达到失败上限
	f.cache.EXPECT().Get(mock.Anything, "login:fail:user:zhangsan").Return("5", nil).Once()

	result, err := f.authService.Login(ctx, req, clientIP)

	assert.Nil(t, result)
	assertBizError(t, err, common.PASSWORD_ENTER_EXCEED_LIMIT)
	assert.Contains(t, err.Error(), "账户已临时锁定")
}

func TestLogin_UsernameNormalization(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()
	clientIP := "192.168.1.100"

	req := &bo.LoginRequest{
		Username:    "  ZhangSan  ",
		Password:    "Admin@2024",
		CaptchaKey:  "key-001",
		CaptchaCode: "123456",
	}

	// 输入 "  ZhangSan  " 经 TrimSpace + ToLower 标准化后应为 "zhangsan"
	// 通过让用户名维度已锁定来间接验证标准化逻辑：
	// 如果标准化正确，会查询 "login:fail:user:zhangsan"（而非原始输入）
	mockCacheGetNotFound(f.cache, "login:fail:ip:"+clientIP)
	f.cache.EXPECT().Get(mock.Anything, "login:fail:user:zhangsan").Return("5", nil).Once()

	result, err := f.authService.Login(ctx, req, clientIP)

	assert.Nil(t, result)
	assertBizError(t, err, common.PASSWORD_ENTER_EXCEED_LIMIT)
	// 关键验证点：mock 期望的 key 是 "login:fail:user:zhangsan"
	// 如果用户名未经标准化（仍为 "  ZhangSan  "），mock 不会匹配，测试将失败
}

// ============================================================
// GetAuthInfo 测试用例
// ============================================================

func TestGetAuthInfo_Success(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	expectedVO := &vo.UserInfoVO{
		UserId:   1001,
		Username: "zhangsan",
		Nickname: "张三",
		Avatar:   "https://cdn.example.com/avatar/zhangsan.png",
		Roles:    []string{"admin"},
		Perms:    []string{"sys:user:list", "sys:role:list"},
	}

	f.userService.EXPECT().GetCurrentUserInfo(ctx, int64(1001)).Return(expectedVO, nil).Once()

	result, err := f.authService.GetAuthInfo(ctx, 1001)

	assert.NoError(t, err)
	assert.Equal(t, expectedVO.UserId, result.UserId)
	assert.Equal(t, "zhangsan", result.Username)
	assert.Equal(t, "张三", result.Nickname)
	assert.Equal(t, []string{"admin"}, result.Roles)
	assert.Equal(t, []string{"sys:user:list", "sys:role:list"}, result.Perms)
}

func TestGetAuthInfo_UserNotExist(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	f.userService.EXPECT().GetCurrentUserInfo(ctx, int64(9999)).
		Return(nil, common.NewBizError(common.USER_NOT_EXIST, "用户不存在")).Once()

	result, err := f.authService.GetAuthInfo(ctx, 9999)

	assert.Nil(t, result)
	assertBizError(t, err, common.USER_NOT_EXIST)
}

// ============================================================
// AddTokenToBlacklist 测试用例
// ============================================================

func TestAddTokenToBlacklist_Success(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	user := newTestUser()
	accessToken, _ := generateTestToken(t, user)

	// 解析 token 获取 jti，用于断言 cache.Set 的 key
	j := newTestJWT()
	claims, err := j.ParseToken(accessToken)
	assert.NoError(t, err)
	expectedKey := common.BlacklistPrefix + claims.ID

	// TTL 使用 mock.Anything：AddTokenToBlacklist 内部计算 time.Until(claims.ExpiresAt.Time)
	// 作为黑名单过期时间（Token 剩余有效期），与 testJWTTTL 存在微小时间差，无法精确匹配
	f.cache.EXPECT().Set(ctx, expectedKey, "1", mock.Anything).Return(nil).Once()

	err = f.authService.AddTokenToBlacklist(ctx, accessToken)

	assert.NoError(t, err)
}

func TestAddTokenToBlacklist_EmptyToken(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	err := f.authService.AddTokenToBlacklist(ctx, "")

	// 空 Token 应直接返回 nil，不触发任何缓存操作
	assert.NoError(t, err)
}

func TestAddTokenToBlacklist_InvalidToken(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	err := f.authService.AddTokenToBlacklist(ctx, "this-is-not-a-valid-jwt-token")

	// 无效 Token 解析失败，应直接返回 nil（无效 Token 无法通过验证，无需加入黑名单）
	assert.NoError(t, err)
}

func TestAddTokenToBlacklist_CacheWriteFailure(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	user := newTestUser()
	accessToken, _ := generateTestToken(t, user)

	j := newTestJWT()
	claims, err := j.ParseToken(accessToken)
	assert.NoError(t, err)
	expectedKey := common.BlacklistPrefix + claims.ID

	cacheErr := errors.New("redis connection refused")
	f.cache.EXPECT().Set(ctx, expectedKey, "1", mock.Anything).Return(cacheErr).Once()

	err = f.authService.AddTokenToBlacklist(ctx, accessToken)

	// 缓存写入失败应透传错误
	assert.ErrorIs(t, err, cacheErr)
}

// ============================================================
// RefreshToken 测试用例
// ============================================================

// TestRefreshToken_EmptyToken 验证空刷新令牌的参数校验
// 业务场景：用户调用刷新接口但未传入 refreshToken
// 预期行为：返回 PARAM_ERROR 错误，提示"刷新令牌不能为空"
func TestRefreshToken_EmptyToken(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	result, err := f.authService.RefreshToken(ctx, "")

	assert.Nil(t, result)
	assertBizError(t, err, common.PARAM_ERROR)
	assert.Contains(t, err.Error(), "刷新令牌不能为空")
}

// TestRefreshToken_InvalidToken 验证无效刷新令牌的解析校验
// 业务场景：用户传入格式错误的 refreshToken（非合法 JWT 格式）
// 预期行为：返回 TOKEN_INVALID 错误，提示"无效的刷新令牌"
func TestRefreshToken_InvalidToken(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	result, err := f.authService.RefreshToken(ctx, "not-a-valid-jwt-token")

	assert.Nil(t, result)
	assertBizError(t, err, common.TOKEN_INVALID)
	assert.Contains(t, err.Error(), "无效的刷新令牌")
}

// TestRefreshToken_TokenInBlacklist 验证已失效刷新令牌的黑名单校验
// 业务场景：用户的 refreshToken 已被注销（加入黑名单）
// 预期行为：返回 TOKEN_INVALID 错误，提示"刷新令牌已失效，请重新登录"
func TestRefreshToken_TokenInBlacklist(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	user := newTestUser()
	_, refreshToken := generateTestToken(t, user)

	j := newTestJWT()
	claims, err := j.ParseToken(refreshToken)
	assert.NoError(t, err)
	expectedKey := common.BlacklistPrefix + claims.ID

	// Token 解析成功，但在黑名单中
	f.cache.EXPECT().Exists(ctx, expectedKey).Return(true, nil).Once()

	result, err := f.authService.RefreshToken(ctx, refreshToken)

	assert.Nil(t, result)
	assertBizError(t, err, common.TOKEN_INVALID)
	assert.Contains(t, err.Error(), "刷新令牌已失效")
}

// TestRefreshToken_UserNotExist 验证刷新令牌对应用户不存在的校验
// 业务场景：refreshToken 解析成功，但用户已被删除
// 预期行为：返回 USER_NOT_EXIST 错误
func TestRefreshToken_UserNotExist(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	user := newTestUser()
	_, refreshToken := generateTestToken(t, user)

	j := newTestJWT()
	claims, err := j.ParseToken(refreshToken)
	assert.NoError(t, err)
	expectedKey := common.BlacklistPrefix + claims.ID

	// Token 不在黑名单
	f.cache.EXPECT().Exists(ctx, expectedKey).Return(false, nil).Once()
	// 用户不存在
	f.userService.EXPECT().GetUserAuthInfo(ctx, "zhangsan").
		Return(nil, common.NewBizError(common.USER_NOT_EXIST, "用户不存在")).Once()

	result, err := f.authService.RefreshToken(ctx, refreshToken)

	assert.Nil(t, result)
	assertBizError(t, err, common.USER_NOT_EXIST)
}

// TestRefreshToken_UserDisabled 验证刷新令牌对应用户被禁用的校验
// 业务场景：refreshToken 解析成功，用户存在但状态为禁用
// 预期行为：返回 USER_ACCOUNT_LOCKED 错误
func TestRefreshToken_UserDisabled(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	user := newTestUser()
	_, refreshToken := generateTestToken(t, user)

	j := newTestJWT()
	claims, err := j.ParseToken(refreshToken)
	assert.NoError(t, err)
	expectedKey := common.BlacklistPrefix + claims.ID

	// Token 不在黑名单
	f.cache.EXPECT().Exists(ctx, expectedKey).Return(false, nil).Once()

	// 用户存在但被禁用
	disabledUser := &model.UserAuthInfo{
		UserId:   1001,
		Username: "zhangsan",
		Status:   0, // 禁用状态
	}
	f.userService.EXPECT().GetUserAuthInfo(ctx, "zhangsan").Return(disabledUser, nil).Once()

	result, err := f.authService.RefreshToken(ctx, refreshToken)

	assert.Nil(t, result)
	assertBizError(t, err, common.USER_ACCOUNT_LOCKED)
	assert.Contains(t, err.Error(), "用户已被禁用")
}

// TestLogout_Success 验证正常注销流程
// 业务场景：用户请求注销，请求头携带有效 Token
// 预期行为：Token 加入黑名单，清理用户登录状态缓存，返回成功
func TestLogout_Success(t *testing.T) {
	f := setupTest(t)

	user := newTestUser()
	accessToken, _ := generateTestToken(t, user)

	j := newTestJWT()
	claims, err := j.ParseToken(accessToken)
	assert.NoError(t, err)
	expectedKey := common.BlacklistPrefix + claims.ID

	// 模拟 gin.Context
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("POST", "/api/v1/auth/logout", nil)
	c.Request.Header.Set("Authorization", "Bearer "+accessToken)

	// 期望：Token 加入黑名单（TTL 使用 mock.Anything，因为实际使用 Token 剩余有效期）
	f.cache.EXPECT().Set(mock.Anything, expectedKey, "1", mock.Anything).Return(nil).Once()
	// 期望：清理用户登录状态缓存（多端登录互斥场景）
	f.cache.EXPECT().Delete(mock.Anything, "zhangsan").Return(nil).Once()

	err = f.authService.Logout(c)

	assert.NoError(t, err)
}

// ============================================================
// Logout 测试用例
// ============================================================

// TestLogout_NoToken 验证无Token时的注销行为
// 业务场景：用户请求注销但请求头中无 Token
// 预期行为：直接返回 nil，视为已注销
func TestLogout_NoToken(t *testing.T) {
	f := setupTest(t)

	// 模拟 gin.Context
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("POST", "/api/v1/auth/logout", nil)
	// 不设置 Authorization 头

	err := f.authService.Logout(c)

	assert.NoError(t, err)
}

// TestLogout_BlacklistFailure 验证黑名单写入失败时的错误处理
// 业务场景：注销时缓存服务不可用，写入黑名单失败
// 预期行为：返回 SYSTEM_EXECUTION_ERROR 错误
func TestLogout_BlacklistFailure(t *testing.T) {
	f := setupTest(t)

	user := newTestUser()
	accessToken, _ := generateTestToken(t, user)

	j := newTestJWT()
	claims, err := j.ParseToken(accessToken)
	assert.NoError(t, err)
	expectedKey := common.BlacklistPrefix + claims.ID

	// 模拟 gin.Context
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("POST", "/api/v1/auth/logout", nil)
	c.Request.Header.Set("Authorization", "Bearer "+accessToken)

	// 模拟缓存写入失败
	cacheErr := errors.New("redis connection refused")
	f.cache.EXPECT().Set(mock.Anything, expectedKey, "1", mock.Anything).Return(cacheErr).Once()

	err = f.authService.Logout(c)

	assertBizError(t, err, common.SYSTEM_EXECUTION_ERROR)
	assert.Contains(t, err.Error(), "注销失败")
}

// ============================================================
// IsTokenBlacklisted 测试用例
// ============================================================

func TestIsTokenBlacklisted_TokenInBlacklist(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	user := newTestUser()
	accessToken, _ := generateTestToken(t, user)

	j := newTestJWT()
	claims, err := j.ParseToken(accessToken)
	assert.NoError(t, err)
	expectedKey := common.BlacklistPrefix + claims.ID

	f.cache.EXPECT().Exists(ctx, expectedKey).Return(true, nil).Once()

	result := f.authService.IsTokenBlacklisted(ctx, accessToken)

	assert.True(t, result, "在黑名单中的 Token 应返回 true")
}

func TestIsTokenBlacklisted_TokenNotInBlacklist(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	user := newTestUser()
	accessToken, _ := generateTestToken(t, user)

	j := newTestJWT()
	claims, err := j.ParseToken(accessToken)
	assert.NoError(t, err)
	expectedKey := common.BlacklistPrefix + claims.ID

	f.cache.EXPECT().Exists(ctx, expectedKey).Return(false, nil).Once()

	result := f.authService.IsTokenBlacklisted(ctx, accessToken)

	assert.False(t, result, "不在黑名单中的 Token 应返回 false")
}

func TestIsTokenBlacklisted_EmptyToken(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	result := f.authService.IsTokenBlacklisted(ctx, "")

	assert.False(t, result, "空 Token 应返回 false")
}

func TestIsTokenBlacklisted_CacheError_ReturnFalse(t *testing.T) {
	f := setupTest(t)
	ctx := context.Background()

	user := newTestUser()
	accessToken, _ := generateTestToken(t, user)

	j := newTestJWT()
	claims, err := j.ParseToken(accessToken)
	assert.NoError(t, err)
	expectedKey := common.BlacklistPrefix + claims.ID

	// 缓存异常时应保证服务可用性，返回 false 而非 panic/error
	f.cache.EXPECT().Exists(ctx, expectedKey).Return(false, errors.New("redis timeout")).Once()

	result := f.authService.IsTokenBlacklisted(ctx, accessToken)

	assert.False(t, result, "缓存异常时应容错返回 false，保证服务可用性")
}

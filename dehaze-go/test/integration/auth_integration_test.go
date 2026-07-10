package integration

import (
	"encoding/json"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	tu "github.com/earthyzinc/dehaze-go/test/integration/testutil"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ============================================================
// 测试环境初始化
// ============================================================

func TestMain(m *testing.M) {
	tu.InitTestEnv("../../")
	tu.Engine = tu.SetupAuthRouter()
	tu.CleanLoginFailCounts("admin", "test")

	code := m.Run()
	os.Exit(code)
}

// ============================================================
// 1. 验证码链路
// ============================================================

func TestCaptcha_GetSuccess(t *testing.T) {
	w := tu.DoRequest(http.MethodGet, "/api/v1/auth/captcha", nil, "")

	assert.Equal(t, http.StatusOK, w.Code)
	resp := tu.ParseResponse(t, w)
	assert.Equal(t, common.SUCCESS.Code, resp.Code)
	assert.Equal(t, "验证码获取成功", resp.Msg)

	var data struct {
		CaptchaKey    string `json:"captchaKey"`
		CaptchaBase64 string `json:"captchaBase64"`
	}
	require.NoError(t, json.Unmarshal(resp.Data, &data))
	assert.NotEmpty(t, data.CaptchaKey, "captchaKey 不应为空")
	assert.NotEmpty(t, data.CaptchaBase64, "captchaBase64 不应为空")
	assert.Contains(t, data.CaptchaBase64, "data:image/png;base64,", "captchaBase64 应为 base64 图片格式")
}

func TestCaptcha_Expired(t *testing.T) {
	captchaKey := tu.GetCaptcha(t)

	// 从缓存中删除验证码，模拟过期
	tu.DeleteCaptchaFromCache(captchaKey)

	loginBody := map[string]string{
		"username":    "admin",
		"password":    "123456",
		"captchaKey":  captchaKey,
		"captchaCode": "000000",
	}
	w := tu.DoRequest(http.MethodPost, "/api/v1/auth/login", loginBody, "")

	assert.Equal(t, http.StatusOK, w.Code)
	resp := tu.ParseResponse(t, w)
	assert.Equal(t, common.VERIFY_CODE_ERROR.Code, resp.Code, "验证码过期应返回验证码错误码")
}

// ============================================================
// 2. 登录成功链路（验证码正确 + 登录成功）
// ============================================================

func TestLogin_Success_Admin(t *testing.T) {
	captchaKey := tu.GetCaptcha(t)
	testCode := "999999"
	tu.InjectCaptchaAnswer(t, captchaKey, testCode)

	loginBody := map[string]string{
		"username":    "admin",
		"password":    "123456",
		"captchaKey":  captchaKey,
		"captchaCode": testCode,
	}
	w := tu.DoRequest(http.MethodPost, "/api/v1/auth/login", loginBody, "")

	assert.Equal(t, http.StatusOK, w.Code)
	resp := tu.ParseResponse(t, w)
	assert.Equal(t, common.SUCCESS.Code, resp.Code)
	assert.Equal(t, "登录成功", resp.Msg)

	var loginData struct {
		AccessToken  string `json:"accessToken"`
		TokenType    string `json:"tokenType"`
		RefreshToken string `json:"refreshToken"`
		Expires      int64  `json:"expires"`
	}
	require.NoError(t, json.Unmarshal(resp.Data, &loginData))

	assert.NotEmpty(t, loginData.AccessToken, "accessToken 不应为空")
	assert.Equal(t, "Bearer", loginData.TokenType, "tokenType 应为 Bearer")
	assert.NotEmpty(t, loginData.RefreshToken, "refreshToken 不应为空")
	assert.Greater(t, loginData.Expires, time.Now().UnixMilli(), "expires 应大于当前时间戳(毫秒)")
}

func TestLogin_WrongPassword(t *testing.T) {
	captchaKey := tu.GetCaptcha(t)
	testCode := "111111"
	tu.InjectCaptchaAnswer(t, captchaKey, testCode)

	loginBody := map[string]string{
		"username":    "admin",
		"password":    "wrong_password",
		"captchaKey":  captchaKey,
		"captchaCode": testCode,
	}
	w := tu.DoRequest(http.MethodPost, "/api/v1/auth/login", loginBody, "")

	assert.Equal(t, http.StatusOK, w.Code)
	resp := tu.ParseResponse(t, w)
	assert.NotEqual(t, common.SUCCESS.Code, resp.Code, "密码错误不应返回成功")

	tu.CleanLoginFailCounts("admin")
}

func TestLogin_WrongCaptcha(t *testing.T) {
	captchaKey := tu.GetCaptcha(t)
	testCode := "222222"
	tu.InjectCaptchaAnswer(t, captchaKey, testCode)

	loginBody := map[string]string{
		"username":    "admin",
		"password":    "123456",
		"captchaKey":  captchaKey,
		"captchaCode": "000000", // 故意传错误验证码
	}
	w := tu.DoRequest(http.MethodPost, "/api/v1/auth/login", loginBody, "")

	assert.Equal(t, http.StatusOK, w.Code)
	resp := tu.ParseResponse(t, w)
	assert.Equal(t, common.VERIFY_CODE_ERROR.Code, resp.Code, "验证码错误应返回 VERIFY_CODE_ERROR")

	tu.CleanLoginFailCounts("admin")
}

// ============================================================
// 3. JWT 鉴权中间件链路
// ============================================================

func TestJWT_NoToken_AccessProtectedRoute(t *testing.T) {
	w := tu.DoRequest(http.MethodGet, "/api/v1/auth/me", nil, "")

	assert.Equal(t, http.StatusUnauthorized, w.Code, "无Token访问受保护路由应返回401")
	resp := tu.ParseResponse(t, w)
	assert.Equal(t, common.ACCESS_UNAUTHORIZED.Code, resp.Code, "错误码应为 ACCESS_UNAUTHORIZED")
}

func TestJWT_FakeToken_AccessProtectedRoute(t *testing.T) {
	fakeToken := "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmYWtlIiwiZXhwIjoxNjAwMDAwMDAwfQ.invalid_signature"
	w := tu.DoRequest(http.MethodGet, "/api/v1/auth/me", nil, fakeToken)

	assert.Equal(t, http.StatusUnauthorized, w.Code, "伪造Token应返回401")
	resp := tu.ParseResponse(t, w)
	assert.Equal(t, common.ACCESS_UNAUTHORIZED.Code, resp.Code, "错误码应为 ACCESS_UNAUTHORIZED")
}

func TestJWT_ValidToken_GetAuthInfo(t *testing.T) {
	accessToken, _ := tu.LoginAndGetTokens(t, "admin", "123456")

	w := tu.DoRequest(http.MethodGet, "/api/v1/auth/me", nil, accessToken)

	assert.Equal(t, http.StatusOK, w.Code)
	resp := tu.ParseResponse(t, w)
	assert.Equal(t, common.SUCCESS.Code, resp.Code)

	var userInfo struct {
		UserId   int64    `json:"userId"`
		Username string   `json:"username"`
		Nickname string   `json:"nickname"`
		Avatar   string   `json:"avatar"`
		Roles    []string `json:"roles"`
		Perms    []string `json:"perms"`
	}
	require.NoError(t, json.Unmarshal(resp.Data, &userInfo))
	assert.Greater(t, userInfo.UserId, int64(0), "userId 应大于 0")
	assert.Equal(t, "admin", userInfo.Username, "username 应为 admin")
	assert.NotEmpty(t, userInfo.Nickname, "nickname 不应为空")
	assert.NotNil(t, userInfo.Roles, "roles 不应为 nil")
	assert.NotNil(t, userInfo.Perms, "perms 不应为 nil")
}

func TestJWT_ValidToken_TestUser_GetAuthInfo(t *testing.T) {
	accessToken, _ := tu.LoginAndGetTokens(t, "test", "123456")

	w := tu.DoRequest(http.MethodGet, "/api/v1/auth/me", nil, accessToken)

	assert.Equal(t, http.StatusOK, w.Code)
	resp := tu.ParseResponse(t, w)
	assert.Equal(t, common.SUCCESS.Code, resp.Code)

	var userInfo struct {
		UserId   int64    `json:"userId"`
		Username string   `json:"username"`
		Nickname string   `json:"nickname"`
		Roles    []string `json:"roles"`
		Perms    []string `json:"perms"`
	}
	require.NoError(t, json.Unmarshal(resp.Data, &userInfo))
	assert.Greater(t, userInfo.UserId, int64(0), "userId 应大于 0")
	assert.Equal(t, "test", userInfo.Username, "username 应为 test")
	assert.Equal(t, "测试小用户", userInfo.Nickname, "nickname 应为 测试小用户")
}

// ============================================================
// 4. 注销后 Token 失效
// ============================================================

func TestLogout_TokenInvalidated(t *testing.T) {
	accessToken, _ := tu.LoginAndGetTokens(t, "admin", "123456")

	// 先确认 Token 有效
	w := tu.DoRequest(http.MethodGet, "/api/v1/auth/me", nil, accessToken)
	require.Equal(t, http.StatusOK, w.Code)
	resp := tu.ParseResponse(t, w)
	require.Equal(t, common.SUCCESS.Code, resp.Code, "注销前 Token 应有效")

	// 执行注销
	w = tu.DoRequest(http.MethodPost, "/api/v1/auth/logout", nil, accessToken)
	assert.Equal(t, http.StatusOK, w.Code)
	resp = tu.ParseResponse(t, w)
	assert.Equal(t, common.SUCCESS.Code, resp.Code)
	assert.Equal(t, "注销成功", resp.Msg)

	// 注销后使用同一 Token 访问受保护路由应被拒绝
	w = tu.DoRequest(http.MethodGet, "/api/v1/auth/me", nil, accessToken)
	assert.Equal(t, http.StatusUnauthorized, w.Code, "注销后 Token 应失效，返回 401")
	resp = tu.ParseResponse(t, w)
	assert.Equal(t, common.ACCESS_UNAUTHORIZED.Code, resp.Code, "注销后错误码应为 ACCESS_UNAUTHORIZED")
}

// ============================================================
// 5. Refresh 链路
// ============================================================

func TestRefresh_Success(t *testing.T) {
	_, refreshToken := tu.LoginAndGetTokens(t, "admin", "123456")

	refreshBody := map[string]string{
		"refreshToken": refreshToken,
	}
	w := tu.DoRequest(http.MethodPost, "/api/v1/auth/refresh", refreshBody, refreshToken)

	assert.Equal(t, http.StatusOK, w.Code)
	resp := tu.ParseResponse(t, w)
	assert.Equal(t, common.SUCCESS.Code, resp.Code)
	assert.Equal(t, "刷新成功", resp.Msg)

	var newTokens struct {
		AccessToken  string `json:"accessToken"`
		TokenType    string `json:"tokenType"`
		RefreshToken string `json:"refreshToken"`
		Expires      int64  `json:"expires"`
	}
	require.NoError(t, json.Unmarshal(resp.Data, &newTokens))
	assert.NotEmpty(t, newTokens.AccessToken, "新 accessToken 不应为空")
	assert.Equal(t, "Bearer", newTokens.TokenType)
	assert.NotEmpty(t, newTokens.RefreshToken, "新 refreshToken 不应为空")
	assert.Greater(t, newTokens.Expires, time.Now().UnixMilli(), "新 expires 应大于当前时间")

	// 使用新 accessToken 访问受保护路由应成功
	w = tu.DoRequest(http.MethodGet, "/api/v1/auth/me", nil, newTokens.AccessToken)
	assert.Equal(t, http.StatusOK, w.Code)
	meResp := tu.ParseResponse(t, w)
	assert.Equal(t, common.SUCCESS.Code, meResp.Code, "使用刷新后的 accessToken 访问 /me 应成功")
}

func TestRefresh_OldRefreshToken_Invalidated(t *testing.T) {
	_, refreshToken := tu.LoginAndGetTokens(t, "admin", "123456")

	// 第一次刷新应成功
	refreshBody := map[string]string{
		"refreshToken": refreshToken,
	}
	w := tu.DoRequest(http.MethodPost, "/api/v1/auth/refresh", refreshBody, refreshToken)
	require.Equal(t, http.StatusOK, w.Code)
	resp := tu.ParseResponse(t, w)
	require.Equal(t, common.SUCCESS.Code, resp.Code, "首次刷新应成功")

	// 解析首次刷新返回的新 accessToken
	var newTokens struct {
		AccessToken  string `json:"accessToken"`
		RefreshToken string `json:"refreshToken"`
	}
	require.NoError(t, json.Unmarshal(resp.Data, &newTokens))
	require.NotEmpty(t, newTokens.AccessToken, "首次刷新应返回新 accessToken")

	// 使用已被消费的旧 refreshToken 再次刷新应失败
	w = tu.DoRequest(http.MethodPost, "/api/v1/auth/refresh", refreshBody, newTokens.AccessToken)
	assert.Equal(t, http.StatusOK, w.Code)
	resp = tu.ParseResponse(t, w)
	assert.Equal(t, common.TOKEN_INVALID.Code, resp.Code, "旧 refreshToken 应已失效，错误码为 TOKEN_INVALID")
}

func TestRefresh_EmptyRefreshToken(t *testing.T) {
	accessToken, _ := tu.LoginAndGetTokens(t, "admin", "123456")

	refreshBody := map[string]string{
		"refreshToken": "",
	}
	w := tu.DoRequest(http.MethodPost, "/api/v1/auth/refresh", refreshBody, accessToken)

	assert.Equal(t, http.StatusOK, w.Code)
	resp := tu.ParseResponse(t, w)
	assert.NotEqual(t, common.SUCCESS.Code, resp.Code, "空 refreshToken 不应成功")
}

func TestRefresh_FakeRefreshToken(t *testing.T) {
	accessToken, _ := tu.LoginAndGetTokens(t, "admin", "123456")

	refreshBody := map[string]string{
		"refreshToken": "fake.refresh.token",
	}
	w := tu.DoRequest(http.MethodPost, "/api/v1/auth/refresh", refreshBody, accessToken)

	assert.Equal(t, http.StatusOK, w.Code)
	resp := tu.ParseResponse(t, w)
	assert.Equal(t, common.TOKEN_INVALID.Code, resp.Code, "伪造 refreshToken 应返回 TOKEN_INVALID")
}

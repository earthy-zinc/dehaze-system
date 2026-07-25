package integration

import (
	"encoding/json"
	"net/http"
	"os"
	"testing"

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
	assert.Equal(t, common.SUCCESS.Msg, resp.Msg)

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
// 2. 登录成功链路
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
	assert.Equal(t, common.SUCCESS.Msg, resp.Msg)

	var loginData struct {
		SessionID string `json:"sessionId"`
		User      struct {
			ID       int64  `json:"id"`
			Username string `json:"username"`
			Nickname string `json:"nickname"`
		} `json:"user"`
	}
	require.NoError(t, json.Unmarshal(resp.Data, &loginData))

	assert.NotEmpty(t, loginData.SessionID, "sessionId 不应为空")
	assert.Greater(t, loginData.User.ID, int64(0), "user.id 应大于 0")
	assert.Equal(t, "admin", loginData.User.Username, "username 应为 admin")
	assert.NotEmpty(t, loginData.User.Nickname, "nickname 不应为空")
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
		"captchaCode": "000000",
	}
	w := tu.DoRequest(http.MethodPost, "/api/v1/auth/login", loginBody, "")

	assert.Equal(t, http.StatusOK, w.Code)
	resp := tu.ParseResponse(t, w)
	assert.Equal(t, common.VERIFY_CODE_ERROR.Code, resp.Code, "验证码错误应返回 VERIFY_CODE_ERROR")

	tu.CleanLoginFailCounts("admin")
}

// ============================================================
// 3. Session 鉴权中间件链路
// ============================================================

func TestSession_NoSession_AccessProtectedRoute(t *testing.T) {
	w := tu.DoRequest(http.MethodGet, "/api/v1/auth/me", nil, "")

	assert.Equal(t, http.StatusUnauthorized, w.Code, "无Session访问受保护路由应返回401")
	resp := tu.ParseResponse(t, w)
	assert.Equal(t, common.TOKEN_INVALID.Code, resp.Code, "错误码应为 TOKEN_INVALID")
}

func TestSession_FakeSession_AccessProtectedRoute(t *testing.T) {
	fakeSessionID := "00000000-0000-0000-0000-000000000000"
	w := tu.DoRequest(http.MethodGet, "/api/v1/auth/me", nil, fakeSessionID)

	assert.Equal(t, http.StatusUnauthorized, w.Code, "伪造Session应返回401")
	resp := tu.ParseResponse(t, w)
	assert.Equal(t, common.TOKEN_INVALID.Code, resp.Code, "错误码应为 TOKEN_INVALID")
}

func TestSession_Admin_GetAuthInfo(t *testing.T) {
	sessionID := tu.LoginAndGetSessionID(t, "admin", "123456")

	w := tu.DoRequest(http.MethodGet, "/api/v1/auth/me", nil, sessionID)

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

func TestSession_TestUser_GetAuthInfo(t *testing.T) {
	sessionID := tu.LoginAndGetSessionID(t, "test", "123456")

	w := tu.DoRequest(http.MethodGet, "/api/v1/auth/me", nil, sessionID)

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
// 4. 注销后 Session 失效
// ============================================================

func TestLogout_SessionInvalidated(t *testing.T) {
	sessionID := tu.LoginAndGetSessionID(t, "admin", "123456")

	w := tu.DoRequest(http.MethodGet, "/api/v1/auth/me", nil, sessionID)
	require.Equal(t, http.StatusOK, w.Code)
	resp := tu.ParseResponse(t, w)
	require.Equal(t, common.SUCCESS.Code, resp.Code, "注销前 Session 应有效")

	w = tu.DoRequest(http.MethodPost, "/api/v1/auth/logout", nil, sessionID)
	assert.Equal(t, http.StatusOK, w.Code)
	resp = tu.ParseResponse(t, w)
	assert.Equal(t, common.SUCCESS.Code, resp.Code)
	assert.Equal(t, common.SUCCESS.Msg, resp.Msg)

	w = tu.DoRequest(http.MethodGet, "/api/v1/auth/me", nil, sessionID)
	assert.Equal(t, http.StatusUnauthorized, w.Code, "注销后 Session 应失效，返回 401")
	resp = tu.ParseResponse(t, w)
	assert.Equal(t, common.TOKEN_INVALID.Code, resp.Code, "注销后错误码应为 TOKEN_INVALID")
}

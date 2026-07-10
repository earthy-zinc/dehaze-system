package testutil

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/stretchr/testify/require"
)

// ============================================================
// 验证码相关
// ============================================================

// GetCaptcha 请求验证码接口并返回 captchaKey，同时断言接口返回结构正确
func GetCaptcha(t *testing.T) string {
	t.Helper()
	w := DoRequest(http.MethodGet, "/api/v1/auth/captcha", nil, "")
	require.Equal(t, http.StatusOK, w.Code)

	resp := ParseResponse(t, w)
	require.Equal(t, common.SUCCESS.Code, resp.Code, "获取验证码应成功")

	var data struct {
		CaptchaKey    string `json:"captchaKey"`
		CaptchaBase64 string `json:"captchaBase64"`
	}
	require.NoError(t, json.Unmarshal(resp.Data, &data))
	require.NotEmpty(t, data.CaptchaKey, "captchaKey 不应为空")
	require.NotEmpty(t, data.CaptchaBase64, "captchaBase64 不应为空")
	return data.CaptchaKey
}

// InjectCaptchaAnswer 在缓存中直接写入验证码答案，绕过图形识别
func InjectCaptchaAnswer(t *testing.T, captchaKey, answer string) {
	t.Helper()
	store := security.GetCaptchaStore()
	err := store.Set(captchaKey, answer)
	require.NoError(t, err, "注入验证码答案到缓存应成功")
}

// DeleteCaptchaFromCache 从缓存中删除验证码，模拟过期场景
func DeleteCaptchaFromCache(captchaKey string) {
	ctx := context.Background()
	cacheClient := cache.GetCache()
	_ = cacheClient.Delete(ctx, common.CaptchaCodePrefix+captchaKey)
}

// ============================================================
// 登录与 Token 相关
// ============================================================

// LoginAndGetTokens 执行完整登录流程（验证码 -> 登录），返回 accessToken 和 refreshToken
func LoginAndGetTokens(t *testing.T, username, password string) (accessToken, refreshToken string) {
	t.Helper()
	captchaKey := GetCaptcha(t)
	testCaptchaCode := "888888"
	InjectCaptchaAnswer(t, captchaKey, testCaptchaCode)

	loginBody := map[string]string{
		"username":    username,
		"password":    password,
		"captchaKey":  captchaKey,
		"captchaCode": testCaptchaCode,
	}
	w := DoRequest(http.MethodPost, "/api/v1/auth/login", loginBody, "")
	require.Equal(t, http.StatusOK, w.Code)

	resp := ParseResponse(t, w)
	require.Equal(t, common.SUCCESS.Code, resp.Code, "登录应成功, msg=%s", resp.Msg)

	var loginData struct {
		AccessToken  string `json:"accessToken"`
		TokenType    string `json:"tokenType"`
		RefreshToken string `json:"refreshToken"`
		Expires      int64  `json:"expires"`
	}
	require.NoError(t, json.Unmarshal(resp.Data, &loginData))
	require.NotEmpty(t, loginData.AccessToken, "accessToken 不应为空")
	require.NotEmpty(t, loginData.RefreshToken, "refreshToken 不应为空")
	return loginData.AccessToken, loginData.RefreshToken
}

package testutil

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/stretchr/testify/require"
)

// APIResponse 直接复用 common.Response 结构，消除重复定义。
// Data 保留 json.RawMessage 以支持按需反序列化。
type APIResponse struct {
	Code string          `json:"code"`
	Data json.RawMessage `json:"data"`
	Msg  string          `json:"msg"`
}

// DoRequest 构造 HTTP 请求并通过 Engine 执行，返回响应记录器。
// token 非空时自动设置 Authorization: Bearer <token>。
func DoRequest(method, path string, body interface{}, token string) *httptest.ResponseRecorder {
	var reqBody *bytes.Buffer
	if body != nil {
		b, _ := json.Marshal(body)
		reqBody = bytes.NewBuffer(b)
	} else {
		reqBody = bytes.NewBuffer(nil)
	}

	req := httptest.NewRequest(method, path, reqBody)
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	w := httptest.NewRecorder()
	Engine.ServeHTTP(w, req)
	return w
}

// ParseResponse 将 HTTP 响应体解析为 APIResponse。
func ParseResponse(t *testing.T, w *httptest.ResponseRecorder) APIResponse {
	t.Helper()
	var resp APIResponse
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	require.NoError(t, err, "响应体应为合法JSON: %s", w.Body.String())
	return resp
}

// RequireSuccess 断言请求返回 200 且业务码为成功
func RequireSuccess(t *testing.T, w *httptest.ResponseRecorder) APIResponse {
	t.Helper()
	require.Equal(t, http.StatusOK, w.Code)
	resp := ParseResponse(t, w)
	require.Equal(t, common.SUCCESS.Code, resp.Code, "业务码应为成功, msg=%s", resp.Msg)
	return resp
}

// ParseData 将 APIResponse.Data 反序列化到指定结构体
func ParseData(t *testing.T, resp APIResponse, dest interface{}) {
	t.Helper()
	require.NoError(t, json.Unmarshal(resp.Data, dest))
}

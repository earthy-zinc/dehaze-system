package api

// 契约测试模板（Contract Test Template）
//
// 本文件是 dehaze-go API 层"契约测试"的首个模板，供后续路由组（user/member/task...）
// 复制套用。契约测试关注点：HTTP 状态 + 信封 {code, msg, data} 的结构与语义，
// 不断言内部调用序列、不验证 mock 调用次数之外的实现细节。
//
// 模板套路（三固定）：
//  1. 测试 engine 用 gin.New() + ContextErrorHandler() 中间件。
//     原因：Controller 通过 c.Error(err) 把错误挂到 gin.Context.Errors，
//     真正的信封渲染由 ContextErrorHandler 调用 common.HandleError 完成，
//     不挂该中间件则错误无法落到响应体，测试会无法断言信封。
//  2. 单 handler 直接 POST/GET 挂到测试 engine（如 engine.POST("/auth/login", api.Login)），
//     不调用 router 包（其依赖全局 config.GetConfig()，过重且会触发限流等副作用）。
//  3. 依赖用 internal/service/mocks 的 MockIAuthService（已生成），用 EXPECT().Xxx().Return() 注入。
//
// 信封断言维度（每个路由组都应覆盖）：成功 / 参数校验失败 / 业务错误 / 系统异常 / 鉴权 / 边界。

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model/dto"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/service/mocks"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// envelope 是响应信封的通用反序列化结构，专注契约字段而非业务 data 细节。
type envelope struct {
	Code    string          `json:"code"`
	Msg     string          `json:"msg"`
	Data    json.RawMessage `json:"data"`
	TraceId string          `json:"traceId"`
}

// newAuthTestEngine 构造仅挂载错误处理中间件的测试 engine，并挂上目标 handler。
// 不引入 router 包，避免全局 config 依赖与限流副作用。
func newAuthTestEngine(t *testing.T, method, path string, handler gin.HandlerFunc) *gin.Engine {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(middleware.ContextErrorHandler())
	r.Handle(method, path, handler)
	return r
}

// doRequest 统一发起请求并解析信封。
func doRequest(t *testing.T, r *gin.Engine, method, path, contentType string, body []byte) (*httptest.ResponseRecorder, envelope) {
	t.Helper()
	req := httptest.NewRequest(method, path, bytes.NewReader(body))
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	var env envelope
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &env), "响应体必须是合法 JSON 信封")
	return w, env
}

// ============ 成功路径 ============

func TestAuthContract_Login_Success(t *testing.T) {
	authSvc := mocks.NewMockIAuthService(t)
	api := NewAuthApi(authSvc)

	authSvc.EXPECT().Login(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
		Return(&dto.LoginResult{
			SessionID: "sess-123",
			User:      &dto.LoginUser{ID: 1, Username: "alice", Nickname: "Alice"},
		}, nil)

	r := newAuthTestEngine(t, http.MethodPost, "/auth/login", api.Login)

	w, env := doRequest(t, r, http.MethodPost, "/auth/login", "application/json",
		[]byte(`{"username":"alice","password":"secret"}`))

	// 成功契约：HTTP 200 + 成功码 "00000" + data 正确（反序列化断言）
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, common.SUCCESS.Code, env.Code, "成功路径 code 必须为三端统一成功码 00000")
	assert.NotEmpty(t, env.Msg)

	var lr dto.LoginResult
	require.NoError(t, json.Unmarshal(env.Data, &lr))
	assert.Equal(t, "sess-123", lr.SessionID)
	require.NotNil(t, lr.User)
	assert.Equal(t, int64(1), lr.User.ID)
	assert.Equal(t, "alice", lr.User.Username)
}

func TestAuthContract_Captcha_Success(t *testing.T) {
	authSvc := mocks.NewMockIAuthService(t)
	api := NewAuthApi(authSvc)

	authSvc.EXPECT().GetCaptcha(mock.Anything, mock.Anything).
		Return(&dto.CaptchaResult{CaptchaKey: "key-1", CaptchaBase64: "data:img"}, nil)

	r := newAuthTestEngine(t, http.MethodGet, "/auth/captcha", api.Captcha)

	w, env := doRequest(t, r, http.MethodGet, "/auth/captcha?_=1", "", nil)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, common.SUCCESS.Code, env.Code)
	assert.NotEmpty(t, env.Msg)

	var cr dto.CaptchaResult
	require.NoError(t, json.Unmarshal(env.Data, &cr))
	assert.Equal(t, "key-1", cr.CaptchaKey)
	assert.Contains(t, cr.CaptchaBase64, "data:img")
}

// ============ 参数校验失败 ============

func TestAuthContract_Login_MissingRequired(t *testing.T) {
	authSvc := mocks.NewMockIAuthService(t)
	api := NewAuthApi(authSvc)

	r := newAuthTestEngine(t, http.MethodPost, "/auth/login", api.Login)

	// 缺 username/password 两个必填项，ShouldBind 返回 validator 错误
	w, env := doRequest(t, r, http.MethodPost, "/auth/login", "application/json",
		[]byte(`{}`))

	assert.Equal(t, http.StatusOK, w.Code) // 信封统一 200，错误在 code 表达
	assert.NotEqual(t, common.SUCCESS.Code, env.Code, "参数校验失败不应返回成功码")
	assert.Equal(t, common.PARAM_ERROR.Code, env.Code, "缺必填项应映射到 A0400 参数错误")
	assert.NotEmpty(t, env.Msg, "msg 应给出校验原因而非空")
}

func TestAuthContract_Register_FieldTooShort(t *testing.T) {
	authSvc := mocks.NewMockIAuthService(t)
	api := NewAuthApi(authSvc)

	r := newAuthTestEngine(t, http.MethodPost, "/auth/register", api.Register)

	// username 要求 min=3，传入 "ab" 触发长度校验失败
	w, env := doRequest(t, r, http.MethodPost, "/auth/register", "application/json",
		[]byte(`{"username":"ab","password":"123456","nickname":"x","captchaCode":"c","captchaKey":"k"}`))

	assert.Equal(t, http.StatusOK, w.Code)
	assert.NotEqual(t, common.SUCCESS.Code, env.Code)
	assert.Equal(t, common.PARAM_ERROR.Code, env.Code)
	assert.NotEmpty(t, env.Msg)
}

// ============ 业务错误（透传） ============

func TestAuthContract_Login_BizErrorPassthrough(t *testing.T) {
	authSvc := mocks.NewMockIAuthService(t)
	api := NewAuthApi(authSvc)

	bizErr := common.NewBizError(common.USERNAME_OR_PASSWORD_ERROR, "用户名或密码错误")
	authSvc.EXPECT().Login(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
		Return(nil, bizErr)

	r := newAuthTestEngine(t, http.MethodPost, "/auth/login", api.Login)

	w, env := doRequest(t, r, http.MethodPost, "/auth/login", "application/json",
		[]byte(`{"username":"alice","password":"wrong"}`))

	assert.Equal(t, http.StatusOK, w.Code)
	// 业务错误码应原样透传到信封，而非被兜底成系统错误
	assert.Equal(t, common.USERNAME_OR_PASSWORD_ERROR.Code, env.Code, "BizError 错误码应透传")
	assert.Equal(t, "用户名或密码错误", env.Msg, "BizError message 应透传")
}

// ============ 系统异常（兜底） ============

func TestAuthContract_Login_SystemError(t *testing.T) {
	authSvc := mocks.NewMockIAuthService(t)
	api := NewAuthApi(authSvc)

	authSvc.EXPECT().Login(mock.Anything, mock.Anything, mock.Anything, mock.Anything).
		Return(nil, errors.New("db connection refused"))

	r := newAuthTestEngine(t, http.MethodPost, "/auth/login", api.Login)

	w, env := doRequest(t, r, http.MethodPost, "/auth/login", "application/json",
		[]byte(`{"username":"alice","password":"secret"}`))

	assert.Equal(t, http.StatusOK, w.Code)
	// 非 BizError 的未知错误被兜底为系统执行错误 B0001，且不对客户端暴露内部信息
	assert.Equal(t, common.SYSTEM_EXECUTION_ERROR.Code, env.Code)
	assert.NotContains(t, env.Msg, "db connection", "系统异常不应泄露内部错误细节")
}

// ============ 鉴权（未认证访问受保护路由） ============

func TestAuthContract_GetAuthInfo_Unauthenticated(t *testing.T) {
	authSvc := mocks.NewMockIAuthService(t)
	api := NewAuthApi(authSvc)

	r := newAuthTestEngine(t, http.MethodGet, "/auth/me", api.GetAuthInfo)

	// 未注入 claims，security.GetUserID 返回 0，handler 抛 A0301 未授权
	w, env := doRequest(t, r, http.MethodGet, "/auth/me", "", nil)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, common.ACCESS_UNAUTHORIZED.Code, env.Code, "未登录访问受保护路由应返回 A0301")
	assert.NotEmpty(t, env.Msg)
}

func TestAuthContract_GetAuthInfo_Authenticated(t *testing.T) {
	authSvc := mocks.NewMockIAuthService(t)
	api := NewAuthApi(authSvc)

	authSvc.EXPECT().GetAuthInfo(mock.Anything, int64(42)).
		Return(&vo.UserInfoVO{UserId: 42, Username: "alice", Nickname: "Alice"}, nil)

	// 鉴权后由上游中间件注入 *security.CustomClaims，此处直接注入模拟已认证上下文
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	engine.Use(middleware.ContextErrorHandler())
	engine.GET("/auth/me", func(c *gin.Context) {
		c.Set("claims", &security.CustomClaims{UserID: 42})
		api.GetAuthInfo(c)
	})

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/auth/me", nil)
	engine.ServeHTTP(w, req)

	var env envelope
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &env))

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, common.SUCCESS.Code, env.Code, "已认证访问应成功")
	var ui vo.UserInfoVO
	require.NoError(t, json.Unmarshal(env.Data, &ui))
	assert.Equal(t, int64(42), ui.UserId)
	assert.Equal(t, "alice", ui.Username)
}

// ============ 边界输入 ============

func TestAuthContract_Login_InvalidJSON(t *testing.T) {
	authSvc := mocks.NewMockIAuthService(t)
	api := NewAuthApi(authSvc)

	r := newAuthTestEngine(t, http.MethodPost, "/auth/login", api.Login)

	// 非法 JSON：ShouldBind 解析失败（json.SyntaxError），确定属于客户端请求格式问题 → A0400
	w, env := doRequest(t, r, http.MethodPost, "/auth/login", "application/json",
		[]byte(`{not-json`))

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, common.PARAM_ERROR.Code, env.Code, "非法 JSON 应映射到 A0400 参数错误而非系统错误")
	assert.NotEmpty(t, env.Msg)
}

func TestAuthContract_Login_WrongContentType(t *testing.T) {
	authSvc := mocks.NewMockIAuthService(t)
	api := NewAuthApi(authSvc)

	r := newAuthTestEngine(t, http.MethodPost, "/auth/login", api.Login)

	// 未声明 Content-Type 且 body 为表单文本，ShouldBind 无法按 JSON 解析 → 绑定失败（A0400）
	w, env := doRequest(t, r, http.MethodPost, "/auth/login", "",
		[]byte(`username=alice&password=secret`))

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, common.PARAM_ERROR.Code, env.Code, "错误 Content-Type 不应成功绑定，应映射 A0400")
	assert.NotEmpty(t, env.Msg)
}

func TestAuthContract_Login_EmptyBody(t *testing.T) {
	authSvc := mocks.NewMockIAuthService(t)
	api := NewAuthApi(authSvc)

	r := newAuthTestEngine(t, http.MethodPost, "/auth/login", api.Login)

	// 空 body（JSON Content-Type）：ShouldBind 读不到字段（io.EOF），确定属于客户端请求格式问题 → A0400
	w, env := doRequest(t, r, http.MethodPost, "/auth/login", "application/json",
		[]byte(``))

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, common.PARAM_ERROR.Code, env.Code, "空 body 应映射 A0400 参数校验而非系统错误")
	assert.NotEmpty(t, env.Msg)
}

func TestAuthContract_Login_TypeMismatch(t *testing.T) {
	authSvc := mocks.NewMockIAuthService(t)
	api := NewAuthApi(authSvc)

	r := newAuthTestEngine(t, http.MethodPost, "/auth/login", api.Login)

	// JSON 类型不匹配：username 字段期望字符串却传入数字，解码触发 json.UnmarshalTypeError → A0400
	w, env := doRequest(t, r, http.MethodPost, "/auth/login", "application/json",
		[]byte(`{"username":123,"password":"secret"}`))

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, common.PARAM_ERROR.Code, env.Code, "JSON 类型不匹配应映射 A0400 参数错误")
	assert.NotEmpty(t, env.Msg)
}

func TestAuthContract_Login_HugeInvalidBody(t *testing.T) {
	authSvc := mocks.NewMockIAuthService(t)
	api := NewAuthApi(authSvc)

	r := newAuthTestEngine(t, http.MethodPost, "/auth/login", api.Login)

	// 超大非法 body：过长的非法 JSON，解码触发 json.SyntaxError → A0400
	huge := `{"username":"` + string(make([]byte, 100000)) + `"}`
	w, env := doRequest(t, r, http.MethodPost, "/auth/login", "application/json",
		[]byte(huge))

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, common.PARAM_ERROR.Code, env.Code, "超大非法 body 应映射 A0400 参数错误")
	assert.NotEmpty(t, env.Msg)
}

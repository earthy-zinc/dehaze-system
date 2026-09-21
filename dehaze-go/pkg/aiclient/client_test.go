package aiclient

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/earthyzinc/dehaze-go/pkg/trace"
	"github.com/gin-gonic/gin"
)

type upstreamCapture struct {
	method  string
	path    string
	query   string
	headers http.Header
	body    string
}

// flushingRecorder 在首次 flush 时发出信号：让测试无需并发读取响应体即可确认流已开始回传
type flushingRecorder struct {
	*httptest.ResponseRecorder
	flushed chan struct{}
}

func newFlushingRecorder() *flushingRecorder {
	return &flushingRecorder{
		ResponseRecorder: httptest.NewRecorder(),
		flushed:          make(chan struct{}, 1),
	}
}

func (r *flushingRecorder) Flush() {
	r.ResponseRecorder.Flush()
	select {
	case r.flushed <- struct{}{}:
	default:
	}
}

func newTestClient(t *testing.T, serviceURL string) *Client {
	t.Helper()
	client, err := New(options.AI{ServiceURL: serviceURL, Timeout: 5, StreamTimeout: 5, ConnectTimeout: 2})
	if err != nil {
		t.Fatalf("创建转发客户端失败: %v", err)
	}
	return client
}

func TestNew_RequiresServiceURL(t *testing.T) {
	if _, err := New(options.AI{}); err == nil {
		t.Fatal("未配置 ai.serviceUrl 时应启动即失败")
	}
}

func TestForwardJSON_RelaysPathQueryBodyIdentityAndResponse(t *testing.T) {
	const upstreamBody = `{"code":"00000","msg":"ok","data":{"taskId":"t-1"}}`
	captured := make(chan upstreamCapture, 1)

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured <- upstreamCapture{r.Method, r.URL.Path, r.URL.RawQuery, r.Header.Clone(), string(body)}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Ai-Source", "python")
		w.WriteHeader(http.StatusCreated)
		_, _ = io.WriteString(w, upstreamBody)
	}))
	defer upstream.Close()

	client := newTestClient(t, upstream.URL)
	engine := gin.New()
	engine.Use(middleware.Trace())
	engine.POST("/api/v1/ai/agents/:id/test", client.ForwardJSON)

	req := httptest.NewRequest(http.MethodPost,
		"/api/v1/ai/agents/12/test?keyword=%E6%B5%8B%E8%AF%95", strings.NewReader(`{"prompt":"hi"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer dhak_test_key")
	req.Header.Set("Cookie", "X-Session-Id=sess-cookie")
	req.Header.Set(middleware.SessionCookieName, "sess-header")
	req.Header.Set("Idempotency-Key", "idem-1234")
	req.Header.Set(trace.HeaderName, "trace-abc")
	req.Header.Set("X-Forwarded-For", "203.0.113.9, 10.0.0.1")
	req.Header.Set("User-Agent", "dehaze-sdk-js/1.0")
	rec := httptest.NewRecorder()
	engine.ServeHTTP(rec, req)

	got := <-captured
	if got.method != http.MethodPost {
		t.Errorf("method 应透传，got %s", got.method)
	}
	if got.path != "/api/v1/ai/agents/12/test" {
		t.Errorf("path 应原样透传，got %s", got.path)
	}
	if got.query != "keyword=%E6%B5%8B%E8%AF%95" {
		t.Errorf("query 应原样透传，got %s", got.query)
	}
	if got.body != `{"prompt":"hi"}` {
		t.Errorf("请求体应原样透传，got %s", got.body)
	}
	if got.headers.Get("Authorization") != "Bearer dhak_test_key" {
		t.Errorf("Authorization 应透传（python 兼容 API 鉴权依赖），got %q", got.headers.Get("Authorization"))
	}
	if got.headers.Get("Cookie") != "X-Session-Id=sess-cookie" {
		t.Errorf("Cookie 应透传（python 读 request.cookies），got %q", got.headers.Get("Cookie"))
	}
	if got.headers.Get(middleware.SessionCookieName) != "sess-header" {
		t.Errorf("X-Session-Id 头应透传，got %q", got.headers.Get(middleware.SessionCookieName))
	}
	if got.headers.Get(trace.HeaderName) != "trace-abc" {
		t.Errorf("链路头应透传，got %q", got.headers.Get(trace.HeaderName))
	}
	// python 用 x-forwarded-for 记审计 IP/按 IP 限流：只带真实客户端 IP，不整条链透传
	if xff := got.headers.Get("X-Forwarded-For"); xff != "203.0.113.9" {
		t.Errorf("X-Forwarded-For 应为真实客户端 IP，got %q", xff)
	}
	if ua := got.headers.Get("User-Agent"); ua != "dehaze-sdk-js/1.0" {
		t.Errorf("User-Agent 应透传（python 记日志），got %q", ua)
	}
	if ct := got.headers.Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type 应透传，got %q", ct)
	}
	// python 对「发送消息(SSE)」「创建任务」用 Header(alias="Idempotency-Key") 必传，
	// 不透传会被判缺参返回 400（SDK 集成 billing beforeAll「SSE request failed: 400」）
	if ik := got.headers.Get("Idempotency-Key"); ik != "idem-1234" {
		t.Errorf("Idempotency-Key 应透传（python 上游必传头），got %q", ik)
	}

	if rec.Code != http.StatusCreated {
		t.Errorf("上游状态码应原样回传，got %d", rec.Code)
	}
	if rec.Body.String() != upstreamBody {
		t.Errorf("响应体应原样回传（不解包重包），got %s", rec.Body.String())
	}
	if rec.Header().Get("X-Ai-Source") != "python" {
		t.Errorf("上游响应头应回传，got %q", rec.Header().Get("X-Ai-Source"))
	}
}

func TestForwardJSON_UpstreamUnavailableReturnsBusinessEnvelope(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	serviceURL := upstream.URL
	upstream.Close() // 上游不可达

	client := newTestClient(t, serviceURL)
	engine := gin.New()
	engine.POST("/api/v1/ai/models/:id/test", client.ForwardJSON)

	rec := httptest.NewRecorder()
	engine.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/v1/ai/models/3/test", strings.NewReader(`{}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("上游不可达必须返回统一业务信封（HTTP 200），got %d", rec.Code)
	}
	var envelope struct {
		Code string `json:"code"`
		Msg  string `json:"msg"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("响应不是 JSON 信封: %s", rec.Body.String())
	}
	if envelope.Code != common.CALL_THIRD_PARTY_SERVICE_ERROR.Code {
		t.Errorf("业务码应为 %s，got %s", common.CALL_THIRD_PARTY_SERVICE_ERROR.Code, envelope.Code)
	}
	if envelope.Msg == "" {
		t.Error("业务信封应带可读提示")
	}
}

func TestForwardMultipart_StreamsBodyAndKeepsBoundary(t *testing.T) {
	var form bytes.Buffer
	writer := multipart.NewWriter(&form)
	part, err := writer.CreateFormFile("file", "doc.txt")
	if err != nil {
		t.Fatalf("构造 multipart 失败: %v", err)
	}
	if _, err := part.Write([]byte("知识库文档内容")); err != nil {
		t.Fatalf("写入 multipart 失败: %v", err)
	}
	_ = writer.Close()

	captured := make(chan upstreamCapture, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured <- upstreamCapture{r.Method, r.URL.Path, r.URL.RawQuery, r.Header.Clone(), string(body)}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"code":"00000","msg":"ok","data":{"documentId":5}}`)
	}))
	defer upstream.Close()

	client := newTestClient(t, upstream.URL)
	engine := gin.New()
	engine.POST("/api/v1/kb/:id/documents", client.ForwardMultipart)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/kb/3/documents", bytes.NewReader(form.Bytes()))
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set(middleware.SessionCookieName, "sess-1")
	rec := httptest.NewRecorder()
	engine.ServeHTTP(rec, req)

	got := <-captured
	if got.body != form.String() {
		t.Error("multipart 请求体应逐字节透传（不解析、不重编码）")
	}
	if got.headers.Get("Content-Type") != writer.FormDataContentType() {
		t.Errorf("Content-Type（含 boundary）应透传，got %q", got.headers.Get("Content-Type"))
	}
	if got.headers.Get(middleware.SessionCookieName) != "sess-1" {
		t.Error("上传端点的身份头应透传")
	}
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"documentId":5`) {
		t.Errorf("上传响应未回传，status=%d body=%s", rec.Code, rec.Body.String())
	}
}

func TestForwardSSE_RelaysStreamChunks(t *testing.T) {
	const first = "data: {\"delta\":\"你\"}\n\n"
	const second = "data: {\"done\":true}\n\n"

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		for _, chunk := range []string{first, second} {
			_, _ = io.WriteString(w, chunk)
			flusher.Flush()
		}
	}))
	defer upstream.Close()

	client := newTestClient(t, upstream.URL)
	engine := gin.New()
	engine.POST("/api/v1/ai/conversations/:id/messages", client.ForwardSSE)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/ai/conversations/9/messages",
		strings.NewReader(`{"content":"hi"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(middleware.SessionCookieName, "sess-1")
	rec := httptest.NewRecorder()
	engine.ServeHTTP(rec, req)

	if rec.Body.String() != first+second {
		t.Errorf("SSE 事件应逐块原样回传，got %q", rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("SSE Content-Type 应回传，got %q", ct)
	}
	if cc := rec.Header().Get("Cache-Control"); cc != "no-cache" {
		t.Errorf("Cache-Control 应回传，got %q", cc)
	}
}

func TestForwardSSE_CancelsUpstreamOnClientDisconnect(t *testing.T) {
	upstreamCanceled := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		for {
			select {
			case <-r.Context().Done():
				close(upstreamCanceled)
				return
			default:
			}
			if _, err := io.WriteString(w, "data: ping\n\n"); err != nil {
				return
			}
			flusher.Flush()
			time.Sleep(10 * time.Millisecond)
		}
	}))
	defer upstream.Close()

	client := newTestClient(t, upstream.URL)
	client.streamIdle = 30 * time.Second // 排除空闲超时干扰，只有客户端断连能触发取消
	engine := gin.New()
	engine.POST("/api/v1/ai/conversations/:id/messages", client.ForwardSSE)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/ai/conversations/9/messages",
		strings.NewReader(`{"content":"hi"}`)).WithContext(ctx)
	rec := newFlushingRecorder()
	done := make(chan struct{})
	go func() {
		engine.ServeHTTP(rec, req)
		close(done)
	}()

	select {
	case <-rec.flushed:
	case <-time.After(3 * time.Second):
		t.Fatal("SSE 首个事件未回传")
	}

	cancel() // 模拟客户端断连
	select {
	case <-upstreamCanceled:
	case <-time.After(3 * time.Second):
		t.Fatal("客户端断连后未取消上游请求")
	}
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("客户端断连后转发 handler 未退出")
	}
}

func TestForwardSSE_AbortsOnUpstreamIdleTimeout(t *testing.T) {
	upstreamCanceled := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		// 只发响应头，此后长时间不产生事件
		<-r.Context().Done()
		close(upstreamCanceled)
	}))
	defer upstream.Close()

	client := newTestClient(t, upstream.URL)
	client.streamIdle = 200 * time.Millisecond // 同包构造短超时，避免测试等待配置默认的 300s
	engine := gin.New()
	engine.POST("/api/v1/ai/conversations/:id/messages", client.ForwardSSE)

	done := make(chan struct{})
	go func() {
		engine.ServeHTTP(httptest.NewRecorder(),
			httptest.NewRequest(http.MethodPost, "/api/v1/ai/conversations/9/messages",
				strings.NewReader(`{"content":"hi"}`)))
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("上游空闲时应按空闲超时中断转发")
	}
	select {
	case <-upstreamCanceled:
	case <-time.After(3 * time.Second):
		t.Fatal("空闲超时后未取消上游请求")
	}
}

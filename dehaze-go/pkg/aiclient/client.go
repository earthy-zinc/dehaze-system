// Package aiclient 把 AI 域 B 类端点（强依赖 deepagents/LLM/ES）转发给 dehaze-python。
// AI 行为只有一份实现（dehaze-python/app/router），Go 端不做业务逻辑，只做身份透传与报文搬运：
// 三端共享 Redis session，透传 Authorization/Cookie/X-Session-Id 后 python 鉴权零改动。
package aiclient

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/earthyzinc/dehaze-go/pkg/trace"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

const (
	// 缓冲模式的请求体上限：JSON 转发只服务于对话/测试/检索类端点，32MB 足够；
	// multipart 走流式，不受此限制（上传大文件不落盘、不进内存）。
	maxBufferedRequest = 32 << 20
	// 缓冲模式的响应体上限：上传/检索等端点返回的是元数据 JSON。
	maxBufferedResponse = 32 << 20
	// 转发读取上游响应的分片大小
	relayChunkSize = 32 << 10
	// 转发 WebSocket 帧的单帧上限：上行是 PCM 音频块、下行是识别结果 JSON，远小于此值
	maxWSMessage = 4 << 20
	// 透传关闭帧的写超时
	wsWriteWait = 5 * time.Second
)

// wsUpgrader 下游（客户端 → Go）握手升级器；来源放开与 python、java 一致，CORS 由网关统一处理
var wsUpgrader = websocket.Upgrader{
	ReadBufferSize:  4096,
	WriteBufferSize: 4096,
	CheckOrigin:     func(*http.Request) bool { return true },
}

// identityHeaders 必须原样透传的身份头。
// Cookie 与 X-Session-Id 覆盖 python 两种会话读取方式（request.cookies / header），
// x-api-key 供 OpenAI/Claude 兼容协议的 API Key 认证使用。
var identityHeaders = []string{"Authorization", "Cookie", middleware.SessionCookieName, "x-api-key"}

// businessHeaders 上游业务侧必传头：与身份无关，但漏传会让 python 直接判"缺参"。
// `Idempotency-Key` 在 python 是 Header(alias=...) 必传（router/ai_conversation.py:207 发送消息 SSE、
// router/task.py:106 创建任务），缺失由校验异常 handler 统一返回 400 + A0400；
// 不透传会让 SDK 集成里的「发送消息」在 python 侧被拒（表现为 billing beforeAll「SSE request failed: 400」，
// 连带整个 billing 套件 skip）。
var businessHeaders = []string{"Idempotency-Key"}

// skipResponseHeaders 属逐跳头或由本地连接自行决定（长度/编码），不能从上游回传
var skipResponseHeaders = map[string]bool{
	"Connection":          true,
	"Keep-Alive":          true,
	"Proxy-Authenticate":  true,
	"Proxy-Authorization": true,
	"Te":                  true,
	"Trailer":             true,
	"Transfer-Encoding":   true,
	"Upgrade":             true,
	"Content-Length":      true,
	"Content-Encoding":    true,
}

var errBodyTooLarge = fmt.Errorf("请求体超过 %dMB 上限", maxBufferedRequest>>20)

// Client 转发客户端（AI 域 B 类端点 + 语音域；行为唯一实现在 dehaze-python）
type Client struct {
	baseURL      string
	jsonClient   *http.Client // 有总超时：普通 JSON 端点
	streamClient *http.Client // 无总超时：SSE 流式与 multipart 大包，靠空闲看门狗/服务端读写超时兜底
	streamIdle   time.Duration
	// wsDialer 上游 WebSocket 拨号器：刻意不设 HandshakeTimeout——python 的 /ws/asr 在 ASR
	// 引擎冷加载完成后才 accept（20~40s，首次连接），握手超时会误杀首次连接
	wsDialer websocket.Dialer
}

// New 创建转发客户端；ServiceURL 缺失时启动即失败，避免延迟到首次调用才暴露配置错误
func New(cfg options.AI) (*Client, error) {
	if strings.TrimSpace(cfg.ServiceURL) == "" {
		return nil, errors.New("AI 服务配置 ai.serviceUrl 未配置，无法创建 AI 转发客户端")
	}

	timeout := secondsOrDefault(cfg.Timeout, 120*time.Second)
	connectTimeout := secondsOrDefault(cfg.ConnectTimeout, 5*time.Second)
	streamIdle := secondsOrDefault(cfg.StreamTimeout, 300*time.Second)

	// 两类客户端共用同一连接池：转发流量大，需要复用 TCP 连接
	transport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   connectTimeout,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   50,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   5 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		ResponseHeaderTimeout: timeout,
	}

	return &Client{
		baseURL:      strings.TrimRight(cfg.ServiceURL, "/"),
		jsonClient:   &http.Client{Timeout: timeout, Transport: transport},
		streamClient: &http.Client{Transport: transport},
		streamIdle:   streamIdle,
		wsDialer: websocket.Dialer{
			NetDialContext: (&net.Dialer{Timeout: connectTimeout}).DialContext,
		},
	}, nil
}

// ForwardJSON 转发普通 JSON 端点：请求体与响应体按上限缓冲，响应体原样回传（不解包重包）
func (c *Client) ForwardJSON(gc *gin.Context) {
	body, ok := readBufferedBody(gc)
	if !ok {
		return
	}
	resp, err := c.do(gc, c.jsonClient, bytes.NewReader(body), int64(len(body)))
	if err != nil {
		c.writeUpstreamError(gc, err)
		return
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxBufferedResponse))
	if err != nil {
		c.writeUpstreamError(gc, fmt.Errorf("读取上游响应失败: %w", err))
		return
	}
	relayResponseHeaders(gc, resp.Header)
	gc.Data(resp.StatusCode, resp.Header.Get("Content-Type"), respBody)
}

// ForwardMultipart 转发文件上传端点：请求体直接流式转发（不落盘、不整体读入内存），
// Content-Type 含 boundary 原样透传。
func (c *Client) ForwardMultipart(gc *gin.Context) {
	resp, err := c.do(gc, c.streamClient, gc.Request.Body, gc.Request.ContentLength)
	if err != nil {
		c.writeUpstreamError(gc, err)
		return
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxBufferedResponse))
	if err != nil {
		c.writeUpstreamError(gc, fmt.Errorf("读取上游响应失败: %w", err))
		return
	}
	relayResponseHeaders(gc, resp.Header)
	gc.Data(resp.StatusCode, resp.Header.Get("Content-Type"), respBody)
}

// ForwardSSE 转发流式端点（SSE 及可能返回 SSE 的 a2a JSON-RPC）：
// 逐块回写并立即 flush，客户端断连（请求 context 取消）即中断上游请求；
// 上游长时间无字节则按空闲超时断开，避免连接被挂死。
func (c *Client) ForwardSSE(gc *gin.Context) {
	body, ok := readBufferedBody(gc)
	if !ok {
		return
	}
	upstreamCtx, cancelUpstream := context.WithCancel(gc.Request.Context())
	defer cancelUpstream()

	req, err := c.buildRequest(gc, upstreamCtx, bytes.NewReader(body), int64(len(body)))
	if err != nil {
		c.writeUpstreamError(gc, err)
		return
	}
	// 显式禁用压缩：流式响应一旦被 gzip 包裹，逐块 flush 会被压缩缓冲吞掉，客户端看不到实时事件
	req.Header.Set("Accept-Encoding", "identity")

	resp, err := c.streamClient.Do(req)
	if err != nil {
		c.writeUpstreamError(gc, err)
		return
	}
	defer resp.Body.Close()

	relayResponseHeaders(gc, resp.Header)
	gc.Writer.WriteHeader(resp.StatusCode)
	gc.Writer.Flush()

	idle := time.AfterFunc(c.streamIdle, cancelUpstream)
	defer idle.Stop()

	buf := make([]byte, relayChunkSize)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			idle.Reset(c.streamIdle)
			if _, writeErr := gc.Writer.Write(buf[:n]); writeErr != nil {
				// 客户端已断开，请求 context 取消会同步中断上游
				cancelUpstream()
				return
			}
			gc.Writer.Flush()
		}
		if readErr != nil {
			c.logStreamAbort(gc, readErr)
			return
		}
	}
}

// do 按给定请求体发起上游请求
func (c *Client) do(gc *gin.Context, client *http.Client, body io.Reader, contentLength int64) (*http.Response, error) {
	req, err := c.buildRequest(gc, gc.Request.Context(), body, contentLength)
	if err != nil {
		return nil, err
	}
	return client.Do(req)
}

// buildRequest 按原 method/path/query 组装上游请求，并透传身份头与链路头
func (c *Client) buildRequest(gc *gin.Context, ctx context.Context, body io.Reader, contentLength int64) (*http.Request, error) {
	in := gc.Request
	req, err := http.NewRequestWithContext(ctx, in.Method, c.upstreamTarget(in), body)
	if err != nil {
		return nil, fmt.Errorf("创建上游请求失败: %w", err)
	}
	if contentLength >= 0 {
		req.ContentLength = contentLength
	}
	req.Header = forwardHeaders(ctx, in)
	if ct := in.Header.Get("Content-Type"); ct != "" {
		req.Header.Set("Content-Type", ct)
	}
	if accept := in.Header.Get("Accept"); accept != "" {
		req.Header.Set("Accept", accept)
	}
	return req, nil
}

// upstreamTarget 上游绝对地址：python 路由与 Go 路由同路径同 query，只换 host
func (c *Client) upstreamTarget(in *http.Request) string {
	target := c.baseURL + in.URL.Path
	if in.URL.RawQuery != "" {
		target += "?" + in.URL.RawQuery
	}
	return target
}

// forwardHeaders 组装转发头：身份头（三端共享 Redis session）+ 业务头 + 链路头。
// 不透传 Host / Content-Length / Connection 等逐跳头，由本地连接自行生成。
func forwardHeaders(ctx context.Context, in *http.Request) http.Header {
	header := http.Header{}
	for _, name := range identityHeaders {
		if value := in.Header.Get(name); value != "" {
			header.Set(name, value)
		}
	}
	for _, name := range businessHeaders {
		if value := in.Header.Get(name); value != "" {
			header.Set(name, value)
		}
	}
	// 链路头优先取中间件解析/生成的结果，客户端已带的同名头由 trace 中间件原样保留
	if traceID := trace.GetTraceID(ctx); traceID != "" {
		header.Set(trace.HeaderName, traceID)
	}
	if traceParent := trace.TraceParentFromContext(ctx); traceParent != "" {
		header.Set(trace.HeaderNameTraceParent, traceParent)
	}
	// python 的 Trace 中间件用 x-forwarded-for 记审计 IP、按 IP 限流，并记 user-agent；
	// 取 Go 侧已解析的客户端 IP（而非原样透传客户端给的整条链）以保持 IP 口径一致
	if ip := trace.IPFromContext(ctx); ip != "" {
		header.Set("X-Forwarded-For", ip)
	}
	if userAgent := trace.UserAgentFromContext(ctx); userAgent != "" {
		header.Set("User-Agent", userAgent)
	}
	return header
}

// ForwardWebSocket 双向透传 WebSocket 连接（语音流式 ASR /ws/asr）。
// 只搬运帧，不解析应用层协议；鉴权事实源在 python——浏览器 WS 握手无法携带自定义头，
// 登录会话凭证经 query 的 sid 传递（三端共享 Redis session，python 侧校验会话归属）。
//
// 先连上游再升级下游：python 的 /ws/asr 在 ASR 引擎就绪后才 accept，先升级下游会让客户端在
// 服务端未就绪时就开始发送 PCM/EOS，需要额外的帧缓冲；上游握手失败时下游尚未升级，
// 可把上游状态码原样透传而不必用关闭码兜底。
func (c *Client) ForwardWebSocket(gc *gin.Context) {
	in := gc.Request
	upstream, resp, err := c.wsDialer.DialContext(gc.Request.Context(), c.wsTarget(in),
		forwardHeaders(gc.Request.Context(), in))
	if err != nil {
		if resp != nil {
			defer resp.Body.Close()
			body, _ := io.ReadAll(io.LimitReader(resp.Body, maxBufferedResponse))
			gc.Data(resp.StatusCode, resp.Header.Get("Content-Type"), body)
			return
		}
		c.writeUpstreamError(gc, fmt.Errorf("连接上游 WebSocket 失败: %w", err))
		return
	}

	downstream, err := wsUpgrader.Upgrade(gc.Writer, in, nil)
	if err != nil {
		_ = upstream.Close()
		logger.WithContext(gc.Request.Context()).Warn("WebSocket 升级失败",
			zap.String("path", in.URL.Path), zap.Error(err))
		return
	}

	// 两个方向各一个 goroutine：各自单向写同一连接，天然无并发写冲突
	done := make(chan struct{}, 2)
	go relayWebSocket(upstream, downstream, done)
	go relayWebSocket(downstream, upstream, done)
	<-done
	// 任一方向结束即关闭两端，让另一方向的阻塞读立即返回
	_ = downstream.Close()
	_ = upstream.Close()
	<-done
}

// wsTarget 上游 WebSocket 绝对地址：(http|https) 基址换算为 (ws|wss)，路径与 query 原样透传
func (c *Client) wsTarget(in *http.Request) string {
	base := c.baseURL
	switch {
	case strings.HasPrefix(base, "https://"):
		base = "wss://" + strings.TrimPrefix(base, "https://")
	case strings.HasPrefix(base, "http://"):
		base = "ws://" + strings.TrimPrefix(base, "http://")
	}
	target := base + in.URL.Path
	if in.URL.RawQuery != "" {
		target += "?" + in.URL.RawQuery
	}
	return target
}

// relayWebSocket 单向搬运 WebSocket 帧；对端关闭时透传关闭码，让两端一起收敛
func relayWebSocket(dst, src *websocket.Conn, done chan<- struct{}) {
	defer func() { done <- struct{}{} }()
	src.SetReadLimit(maxWSMessage)
	for {
		messageType, payload, err := src.ReadMessage()
		if err != nil {
			if closeErr, ok := err.(*websocket.CloseError); ok {
				_ = dst.WriteControl(websocket.CloseMessage,
					websocket.FormatCloseMessage(closeErr.Code, closeErr.Text), time.Now().Add(wsWriteWait))
			}
			return
		}
		if err := dst.WriteMessage(messageType, payload); err != nil {
			return
		}
	}
}

// readBufferedBody 读取请求体；失败时已写响应，返回 ok=false
func readBufferedBody(gc *gin.Context) ([]byte, bool) {
	body, err := io.ReadAll(io.LimitReader(gc.Request.Body, maxBufferedRequest+1))
	if err != nil {
		logger.WithContext(gc.Request.Context()).Error("读取请求体失败",
			zap.String("path", gc.Request.URL.Path), zap.Error(err))
		common.FailWithCodeAndMessage(common.PARAM_ERROR, "读取请求体失败", gc)
		return nil, false
	}
	if len(body) > maxBufferedRequest {
		common.FailWithCodeAndMessage(common.PARAM_ERROR, errBodyTooLarge.Error(), gc)
		return nil, false
	}
	return body, true
}

// relayResponseHeaders 回传上游响应头（跳过逐跳头与长度/编码头，由本地连接决定）
func relayResponseHeaders(gc *gin.Context, header http.Header) {
	for name, values := range header {
		if skipResponseHeaders[http.CanonicalHeaderKey(name)] {
			continue
		}
		for _, value := range values {
			gc.Writer.Header().Add(name, value)
		}
	}
}

// writeUpstreamError 上游不可达/异常时返回统一业务信封，避免裸 500
func (c *Client) writeUpstreamError(gc *gin.Context, err error) {
	logger.WithContext(gc.Request.Context()).Error("AI 服务转发失败",
		zap.String("path", gc.Request.URL.Path), zap.Error(err))
	// 客户端已断开时响应无处可送，直接放弃
	if gc.Request.Context().Err() != nil {
		return
	}
	common.FailWithCodeAndMessage(common.CALL_THIRD_PARTY_SERVICE_ERROR, "AI 服务暂不可用，请稍后重试", gc)
}

// logStreamAbort 区分中断原因：客户端断连、空闲超时、上游异常
func (c *Client) logStreamAbort(gc *gin.Context, err error) {
	if errors.Is(err, io.EOF) {
		return
	}
	log := logger.WithContext(gc.Request.Context())
	path := zap.String("path", gc.Request.URL.Path)
	switch {
	case gc.Request.Context().Err() != nil:
		log.Debug("客户端断开，已中断上游 AI 流式响应", path)
	case errors.Is(err, context.Canceled):
		log.Warn("上游 AI 流式响应空闲超时，已断开", path, zap.Duration("idleTimeout", c.streamIdle))
	default:
		log.Error("上游 AI 流式响应中断", path, zap.Error(err))
	}
}

func secondsOrDefault(seconds int, fallback time.Duration) time.Duration {
	if seconds <= 0 {
		return fallback
	}
	return time.Duration(seconds) * time.Second
}

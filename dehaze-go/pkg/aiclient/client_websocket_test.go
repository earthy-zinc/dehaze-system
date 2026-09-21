package aiclient

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

var testUpgrader = websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}

// TestForwardWebSocket_RelaysFramesAndQuery 锁定 /ws/asr 代理的核心契约：
// query 原样透传到上游（sid 是 python 唯一的 WS 鉴权载体，握手头带不了）、
// 文本/二进制帧双向搬运且不改变帧类型（音频 PCM 不能被当 JSON 解包）、关闭帧透传。
func TestForwardWebSocket_RelaysFramesAndQuery(t *testing.T) {
	gin.SetMode(gin.TestMode)

	upstreamURI := make(chan string, 1)
	upstreamSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := testUpgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		upstreamURI <- r.URL.RequestURI()
		// 原样回显：帧类型与内容都能被下游观察到
		for {
			messageType, payload, err := conn.ReadMessage()
			if err != nil {
				return
			}
			if err := conn.WriteMessage(messageType, payload); err != nil {
				return
			}
		}
	}))
	defer upstreamSrv.Close()

	client, err := New(options.AI{ServiceURL: upstreamSrv.URL})
	if err != nil {
		t.Fatalf("创建转发客户端失败: %v", err)
	}
	engine := gin.New()
	engine.GET("/ws/asr", client.ForwardWebSocket)
	proxySrv := httptest.NewServer(engine)
	defer proxySrv.Close()

	conn, _, err := websocket.DefaultDialer.Dial(
		"ws"+strings.TrimPrefix(proxySrv.URL, "http")+"/ws/asr?sessionId=s-1&sid=sid-1", nil)
	if err != nil {
		t.Fatalf("连接代理 WebSocket 失败: %v", err)
	}
	defer conn.Close()

	select {
	case uri := <-upstreamURI:
		if uri != "/ws/asr?sessionId=s-1&sid=sid-1" {
			t.Errorf("上游收到的请求 URI 应原样保留 query，实际 %q", uri)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("上游未收到握手请求")
	}

	pcm := []byte{0x01, 0x02, 0x03, 0xFF}
	if err := conn.WriteMessage(websocket.BinaryMessage, pcm); err != nil {
		t.Fatalf("发送二进制帧失败: %v", err)
	}
	messageType, payload, err := conn.ReadMessage()
	if err != nil {
		t.Fatalf("读取二进制回显失败: %v", err)
	}
	if messageType != websocket.BinaryMessage || string(payload) != string(pcm) {
		t.Errorf("二进制帧应原样回传，实际 type=%d payload=%v", messageType, payload)
	}

	if err := conn.WriteMessage(websocket.TextMessage, []byte("EOS")); err != nil {
		t.Fatalf("发送文本帧失败: %v", err)
	}
	messageType, payload, err = conn.ReadMessage()
	if err != nil {
		t.Fatalf("读取文本回显失败: %v", err)
	}
	if messageType != websocket.TextMessage || string(payload) != "EOS" {
		t.Errorf("文本帧应原样回传，实际 type=%d payload=%v", messageType, payload)
	}

	// 客户端主动关闭：关闭帧透传到上游，任一方向结束都要让两端收敛，不能挂住
	_ = conn.WriteControl(websocket.CloseMessage,
		websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""), time.Now().Add(time.Second))
	_ = conn.SetReadDeadline(time.Now().Add(5 * time.Second))
	if _, _, err := conn.ReadMessage(); err == nil {
		t.Error("关闭帧透传后应读到关闭错误")
	}
}

// TestForwardWebSocket_RelaysUpstreamHandshakeRejection 上游拒绝握手时（python 侧缺 sid /
// 会话过期等）必须把状态码与响应体原样透传，而不是统一降级成"服务不可用"掩盖真实原因。
func TestForwardWebSocket_RelaysUpstreamHandshakeRejection(t *testing.T) {
	gin.SetMode(gin.TestMode)

	upstreamSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte("session required"))
	}))
	defer upstreamSrv.Close()

	client, err := New(options.AI{ServiceURL: upstreamSrv.URL})
	if err != nil {
		t.Fatalf("创建转发客户端失败: %v", err)
	}
	engine := gin.New()
	engine.GET("/ws/asr", client.ForwardWebSocket)

	rec := httptest.NewRecorder()
	engine.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/ws/asr", nil))

	if rec.Code != http.StatusForbidden {
		t.Errorf("上游 403 应原样透传，实际 %d", rec.Code)
	}
	if body := rec.Body.String(); body != "session required" {
		t.Errorf("上游响应体应原样透传，实际 %q", body)
	}
}

package algorithm

import (
	"bytes"
	"cmp"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/cache/protection"
	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/trace"
	"go.uber.org/zap"
)

const maxBodySize = 5 << 20

// Client Python 算法服务 HTTP 客户端 — 生产级实现
// 特性：
//   - HTTP 连接池（复用 TCP 连接，避免握手开销）
//   - 指数退避重试（仅对网络层/5xx 错误重试，业务错误不重试）
//   - 熔断器（复用 protection.Breaker，失败率达阈值后快速失败）
//   - POST 请求携带幂等键，防止重试导致重复处理
type Client struct {
	baseURL    string
	httpClient *http.Client
	breaker    *protection.Breaker
	maxRetry   int
	backoff    time.Duration
	apiKey     string
}

// NewClient 创建算法服务客户端
func NewClient(cfg options.Algorithm) *Client {
	timeout := time.Duration(cfg.Timeout) * time.Second
	if cfg.Timeout <= 0 {
		timeout = 60 * time.Second
	}
	connectTimeout := time.Duration(cfg.ConnectTimeout) * time.Second
	if cfg.ConnectTimeout <= 0 {
		connectTimeout = 5 * time.Second
	}

	// HTTP 连接池 Transport
	transport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   connectTimeout,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:          20,
		MaxIdleConnsPerHost:   10,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   5 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		ResponseHeaderTimeout: timeout,
	}

	maxRetry := max(cfg.MaxRetry, 0)
	backoff := time.Duration(cfg.RetryBackoffMs) * time.Millisecond
	if cfg.RetryBackoffMs <= 0 {
		backoff = 1000 * time.Millisecond
	}

	// 熔断器：复用 protection.Breaker
	var breaker *protection.Breaker
	if cfg.CircuitBreaker.Enabled {
		opts := []protection.BreakerOption{
			protection.WithFailureThreshold(orElse(cfg.CircuitBreaker.FailureThreshold, 0, 5)),
			protection.WithTimeout(time.Duration(orElse(cfg.CircuitBreaker.Timeout, 0, 30)) * time.Second),
			protection.WithMaxRequests(orElse(cfg.CircuitBreaker.MaxRequests, 0, 3)),
		}
		breaker = protection.NewBreaker(opts...)
		logger.Debug("算法客户端熔断器已启用",
			zap.Int("failureThreshold", orElse(cfg.CircuitBreaker.FailureThreshold, 0, 5)),
			zap.Int("timeoutSeconds", orElse(cfg.CircuitBreaker.Timeout, 0, 30)))
	}

	return &Client{
		baseURL: cfg.ServiceURL,
		httpClient: &http.Client{
			Timeout:   timeout,
			Transport: transport,
		},
		breaker:  breaker,
		maxRetry: maxRetry,
		backoff:  backoff,
		apiKey:   cfg.ApiKey,
	}
}

// doPost 通用 POST 请求 — 集成熔断器 + 重试 + 幂等键
func (c *Client) doPost(ctx context.Context, path string, reqBody any, respBody any) error {
	body, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("序列化请求失败: %w", err)
	}
	url := c.baseURL + path

	// 生成幂等键：同一逻辑请求的所有重试共用同一个键，便于下游去重
	idempotencyKey := generateIdempotencyKey()

	// 通过熔断器包装执行（熔断打开时快速失败）
	exec := func() error {
		return c.doPostWithRetry(ctx, url, body, idempotencyKey, respBody, path)
	}
	if c.breaker != nil {
		if err := c.breaker.Execute(exec); err != nil {
			if errors.Is(err, protection.ErrCircuitOpen) {
				return fmt.Errorf("算法服务熔断中，请稍后重试: %w", err)
			}
			return err
		}
		return nil
	}
	return exec()
}

// doGet 通用 GET 请求 — 用于查询任务状态
func (c *Client) doGet(ctx context.Context, path string, respBody any) error {
	url := c.baseURL + path

	exec := func() error {
		return c.doGetWithRetry(ctx, url, respBody)
	}
	if c.breaker != nil {
		if err := c.breaker.Execute(exec); err != nil {
			if errors.Is(err, protection.ErrCircuitOpen) {
				return fmt.Errorf("算法服务熔断中，请稍后重试: %w", err)
			}
			return err
		}
		return nil
	}
	return exec()
}

// doGetWithRetry 带指数退避的重试逻辑（GET 请求）
func (c *Client) doGetWithRetry(ctx context.Context, url string, respBody any) error {
	var lastErr error
	backoff := c.backoff

	for attempt := 0; attempt <= c.maxRetry; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}
			backoff *= 2
		}

		lastErr = c.doSingleGet(ctx, url, respBody)
		if lastErr == nil {
			return nil
		}
		if !isRetryable(lastErr) {
			return lastErr
		}
	}

	if lastErr != nil {
		return fmt.Errorf("算法服务调用失败（已重试 %d 次）: %w", c.maxRetry, lastErr)
	}
	return nil
}

// doSingleGet 执行单次 GET 请求
func (c *Client) doSingleGet(
	ctx context.Context,
	url string,
	respBody any,
) error {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("创建请求失败: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if traceID := trace.GetTraceID(ctx); traceID != "" {
		httpReq.Header.Set(trace.HeaderName, traceID)
	}
	if tp := trace.TraceParentFromContext(ctx); tp != "" {
		httpReq.Header.Set(trace.HeaderNameTraceParent, tp)
	}
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	start := time.Now()
	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		logger.Error("算法服务请求失败",
			zap.String("url", url),
			zap.Duration("elapsed", time.Since(start)),
			zap.Error(err))
		return fmt.Errorf("算法服务请求失败: %w", err)
	}
	defer httpResp.Body.Close()

	respBytes, err := io.ReadAll(io.LimitReader(httpResp.Body, maxBodySize))
	if err != nil {
		return fmt.Errorf("读取响应失败: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		return &httpStatusError{status: httpResp.StatusCode, body: string(respBytes)}
	}

	var apiResp struct {
		Code string          `json:"code"`
		Msg  string          `json:"msg"`
		Data json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(respBytes, &apiResp); err != nil {
		return fmt.Errorf("解析响应失败: %w", err)
	}
	if apiResp.Code != "00000" {
		return &businessError{code: apiResp.Code, msg: apiResp.Msg}
	}

	if err := json.Unmarshal(apiResp.Data, respBody); err != nil {
		return fmt.Errorf("解析响应数据失败: %w", err)
	}

	logger.Debug("算法服务调用成功",
		zap.String("url", url),
		zap.Duration("elapsed", time.Since(start)))
	return nil
}

// doPostWithRetry 带指数退避的重试逻辑
// 重试策略：
//   - 网络层错误（连接超时/拒绝/EOF）→ 可重试
//   - 5xx 服务端错误 → 可重试
//   - 4xx 客户端错误 → 不重试（请求格式错误，重试无意义）
//   - 业务错误（code != "00000"）→ 不重试（业务逻辑错误，重试无意义）
func (c *Client) doPostWithRetry(
	ctx context.Context,
	url string,
	body []byte,
	idempotencyKey string,
	respBody any,
	path string,
) error {
	var lastErr error
	backoff := c.backoff

	for attempt := 0; attempt <= c.maxRetry; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}
			logger.Debug("算法服务重试",
				zap.String("url", url),
				zap.Int("attempt", attempt),
				zap.Duration("backoff", backoff))
			backoff *= 2
		}

		lastErr = c.doSinglePost(ctx, url, body, idempotencyKey, respBody, path)
		if lastErr == nil {
			return nil
		}

		// 不可重试错误：立即返回
		if !isRetryable(lastErr) {
			return lastErr
		}
	}

	if lastErr != nil {
		return fmt.Errorf("算法服务调用失败（已重试 %d 次）: %w", c.maxRetry, lastErr)
	}
	return nil
}

// doSinglePost 执行单次 POST 请求
func (c *Client) doSinglePost(
	ctx context.Context,
	url string,
	body []byte,
	idempotencyKey string,
	respBody any,
	path string,
) error {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("创建请求失败: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	// 幂等键：重试时携带同一键，便于下游去重（防重复处理）
	httpReq.Header.Set("X-Idempotency-Key", idempotencyKey)
	// 透传 traceId 到 Python 算法服务，形成跨服务全链路追踪
	if traceID := trace.GetTraceID(ctx); traceID != "" {
		httpReq.Header.Set(trace.HeaderName, traceID)
	}
	if tp := trace.TraceParentFromContext(ctx); tp != "" {
		httpReq.Header.Set(trace.HeaderNameTraceParent, tp)
	}
	// 使用 API Key 进行 M2M 认证
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	start := time.Now()
	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		logger.Error("算法服务请求失败",
			zap.String("url", url),
			zap.Duration("elapsed", time.Since(start)),
			zap.Error(err))
		return fmt.Errorf("算法服务请求失败: %w", err)
	}
	defer httpResp.Body.Close()

	respBytes, err := io.ReadAll(io.LimitReader(httpResp.Body, maxBodySize))
	if err != nil {
		return fmt.Errorf("读取响应失败: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		logger.Error("算法服务返回错误",
			zap.Int("status", httpResp.StatusCode),
			zap.String("body", string(respBytes)))
		return &httpStatusError{status: httpResp.StatusCode, body: string(respBytes)}
	}

	// Python 端返回格式: {"code":"00000","msg":"...","data":{...}}
	var apiResp struct {
		Code string          `json:"code"`
		Msg  string          `json:"msg"`
		Data json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(respBytes, &apiResp); err != nil {
		return fmt.Errorf("解析响应失败: %w", err)
	}
	if apiResp.Code != "00000" {
		// 业务错误 → 不可重试
		return &businessError{code: apiResp.Code, msg: apiResp.Msg}
	}

	if err := json.Unmarshal(apiResp.Data, respBody); err != nil {
		return fmt.Errorf("解析响应数据失败: %w", err)
	}

	logger.Debug("算法服务调用成功",
		zap.String("path", path),
		zap.Duration("elapsed", time.Since(start)))
	return nil
}

// httpStatusError HTTP 状态码错误
type httpStatusError struct {
	status int
	body   string
}

func (e *httpStatusError) Error() string {
	return fmt.Sprintf("算法服务返回状态码 %d: %s", e.status, e.body)
}

// businessError 业务错误（Python 服务返回 code != "00000"）
type businessError struct {
	code string
	msg  string
}

func (e *businessError) Error() string {
	return fmt.Sprintf("算法服务返回错误: [%s] %s", e.code, e.msg)
}

// isRetryable 判断错误是否可重试
func isRetryable(err error) bool {
	// 业务错误 → 不可重试
	var be *businessError
	if errors.As(err, &be) {
		return false
	}
	// HTTP 状态码错误：5xx 可重试，4xx 不可重试
	var he *httpStatusError
	if errors.As(err, &he) {
		return he.status >= 500
	}
	// 网络层错误（超时、连接拒绝、EOF 等）→ 可重试
	return true
}

// generateIdempotencyKey 生成随机幂等键
// crypto/rand.Read 失败属于系统级故障（熵池耗尽等），直接 panic 避免生成可预测的幂等键
// 破坏下游去重保障
func generateIdempotencyKey() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		panic(fmt.Errorf("生成幂等键失败: %w", err))
	}
	return hex.EncodeToString(b)
}

// orElse 当 v <= zero 时返回 fallback，否则返回 v
func orElse[T cmp.Ordered](v T, zero T, fallback T) T {
	if v <= zero {
		return fallback
	}
	return v
}

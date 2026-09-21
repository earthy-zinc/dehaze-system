package algorithm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/stretchr/testify/require"
)

// TestDoPostBusinessErrorsDoNotTripBreaker **POST** 路径的上游 4xx 不得计入熔断失败统计。
// 现场：SDK 业务负例连发 `400 A0400 图片地址不允许访问内网资源` / `400 A0410 图片来源不能为空`，
// 阈值 5 被业务负例填满 → open(30s) → 之后**正向**预测/评估全部"熔断速败"。
// 该用例同时钉住 classifyBusinessError 必须包住 doPost（曾只包 doGet 而漏 POST，回归即复发）。
func TestDoPostBusinessErrorsDoNotTripBreaker(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"code":"A0400","msg":"图片地址不允许访问内网资源"}`))
	}))
	defer srv.Close()

	client, err := NewClient(options.Algorithm{
		ServiceURL:     srv.URL,
		MaxRetry:       0,
		CircuitBreaker: options.CircuitBreakerConfig{Enabled: true, FailureThreshold: 5, Timeout: 1, MaxRequests: 1},
	})
	require.NoError(t, err)

	for i := 0; i < 8; i++ {
		err := client.doPost(context.Background(), "/api/v1/prediction", map[string]any{"imageUrl": "http://127.0.0.1/x.jpg"}, nil)
		require.Error(t, err)
		require.NotContains(t, err.Error(), "熔断", "第 %d 次业务 400 不应触发熔断", i+1)
	}
}

// TestDoPostInfraErrorsStillTripBreaker 反证：连接层/5xx 仍须计入（熔断器不能被这次修复废掉）
func TestDoPostInfraErrorsStillTripBreaker(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer srv.Close()

	client, err := NewClient(options.Algorithm{
		ServiceURL:     srv.URL,
		MaxRetry:       0,
		CircuitBreaker: options.CircuitBreakerConfig{Enabled: true, FailureThreshold: 3, Timeout: 30, MaxRequests: 1},
	})
	require.NoError(t, err)

	sawOpen := false
	for i := 0; i < 4; i++ {
		err := client.doPost(context.Background(), "/api/v1/prediction", map[string]any{"a": 1}, nil)
		require.Error(t, err)
		if contains(err.Error(), "熔断") {
			sawOpen = true
			break
		}
	}
	require.True(t, sawOpen, "5xx 连发必须触发熔断")
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (func() bool {
		for i := 0; i+len(sub) <= len(s); i++ {
			if s[i:i+len(sub)] == sub {
				return true
			}
		}
		return false
	})()
}

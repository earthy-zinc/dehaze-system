package protection

import (
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestBreakerIgnoresBusinessErrors 业务性错误（上游 4xx）不得计入熔断失败统计：
// 上游返回 4xx 说明它健康可达；若计入失败，业务负例连发会把熔断器打红，
// 随后正向请求全部速败（现场：SDK 连发 5 条 400 后熔断器 open 30s，正向预测/评估全挂）。
func TestBreakerIgnoresBusinessErrors(t *testing.T) {
	b := NewBreaker(WithFailureThreshold(5))

	businessErr := MarkBusiness(errors.New("算法服务返回状态码 400: {\"code\":\"A0400\"}"))
	for i := 0; i < 20; i++ {
		err := b.Execute(func() error { return businessErr })
		require.Error(t, err)
		require.False(t, errors.Is(err, ErrCircuitOpen), "业务错误不应导致熔断（第 %d 次）", i+1)
	}
	require.NoError(t, b.Execute(func() error { return nil }), "业务错误连发后仍应放行正常请求")

	// 反证：同样次数的基础设施错误必须触发熔断（阈值 5）
	infraErr := fmt.Errorf("算法服务请求失败: dial tcp: connection refused")
	for i := 0; i < 5; i++ {
		_ = b.Execute(func() error { return infraErr })
	}
	require.True(t, errors.Is(b.Execute(func() error { return nil }), ErrCircuitOpen), "基础设施错误达阈值必须熔断")
}

// TestMarkBusinessUnwrapKeepsOriginalError 标记后原错误仍可被 errors.As 取回（上层依赖 httpStatusError 判 4xx/5xx）
func TestMarkBusinessUnwrapKeepsOriginalError(t *testing.T) {
	sentinel := errors.New("upstream 400")
	wrapped := MarkBusiness(fmt.Errorf("算法服务返回状态码 400: %w", sentinel))

	require.True(t, IsBusiness(wrapped))
	require.True(t, errors.Is(wrapped, sentinel))
	require.False(t, IsBusiness(errors.New("plain")))
	require.Nil(t, MarkBusiness(nil))
}

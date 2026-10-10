package aidomain

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

// 看板项的无样本字段必须是 JSON null：落成 0 会把「近 24h 无调用」显示成 100% 成功、0ms。
func TestProviderHealthStatVOMarshalsNullMetrics(t *testing.T) {
	b, err := json.Marshal(ProviderHealthStatVO{
		ProviderID:   1,
		ProviderName: "内置本地模型",
		Health:       "healthy",
		CallCount:    0,
		CircuitOpen:  false,
	})
	require.NoError(t, err)
	require.JSONEq(t, `{
		"providerId": 1,
		"providerName": "内置本地模型",
		"health": "healthy",
		"callCount": 0,
		"successRate": null,
		"rate429": null,
		"p95LatencyMs": null,
		"circuitOpen": false
	}`, string(b))
}

package ai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

// 跨端互认基线：三端共用同一组固定向量（密钥 crossEndTestKey、IV 固定，密文由
// python aes_cipher 生成）。用例自行注入该密钥，不读仓库 .env 的实际密钥，
// 因此密钥轮换不会让用例失真。Java 侧同向量见 AiCredentialCipherTest。
const (
	crossEndTestKey  = "dehaze-cross-end-test-key"
	pythonPlaintext  = "sk-test-cross-end-key-1234"
	pythonCiphertext = "AAECAwQFBgcICQoLDA0ODz1AYgtqdeBmh/6NKWzlW3NCUUnNiiaerx4P+CPjrzaD"
	pythonKeyHash    = "856515f37e98ccb7c9b21fdeaf1960856a9216fe82dcaac672849580b1e0f329"
	pythonMaskedKey  = "sk-test-..."
)

func TestDecryptSecretReadsPythonCiphertext(t *testing.T) {
	t.Setenv("AI_PROVIDER_KEY_ENCRYPTION_KEY", crossEndTestKey)
	plain, err := DecryptSecret(pythonCiphertext)
	require.NoError(t, err)
	require.Equal(t, pythonPlaintext, plain, "Go 必须能解开 python 写入的密文（跨端互认）")
}

func TestEncryptSecretRoundTrip(t *testing.T) {
	t.Setenv("AI_PROVIDER_KEY_ENCRYPTION_KEY", crossEndTestKey)
	cipherText, err := EncryptSecret(pythonPlaintext)
	require.NoError(t, err)
	require.NotEqual(t, pythonPlaintext, cipherText)

	plain, err := DecryptSecret(cipherText)
	require.NoError(t, err)
	require.Equal(t, pythonPlaintext, plain)
}

func TestHashAndMaskSecretAlignPython(t *testing.T) {
	require.Equal(t, pythonKeyHash, HashSecret(pythonPlaintext), "SHA256 查重哈希需与 python 一致")
	require.Equal(t, pythonMaskedKey, MaskSecret(pythonPlaintext), "前缀脱敏需与 python mask_key 一致")
	require.Equal(t, "short...", MaskSecret("short"))
}

// 无样本快照：状态位与调用量确定，比率与延迟必须落成 JSON null（与 python 快照同口径），
// 落成 0/1.0 会让看板把「无数据」显示成 100% 成功、0ms。
func TestNoSampleSnapshotKeepsNullMetrics(t *testing.T) {
	b, err := json.Marshal(noSampleSnapshot("healthy", false))
	require.NoError(t, err)
	require.JSONEq(t, `{
		"status": "healthy",
		"circuit_open": false,
		"total_calls_24h": 0,
		"success_rate": null,
		"error_rate": null,
		"limit_rate": null,
		"p95_latency_ms": null
	}`, string(b))
}

func TestSpeedTierOf(t *testing.T) {
	cases := []struct {
		name     string
		snapshot map[string]any
		want     string
	}{
		{"miss", map[string]any{}, "unknown"},
		{"nil", map[string]any{"p95_latency_ms": nil}, "unknown"},
		{"fast", map[string]any{"p95_latency_ms": float64(1200)}, "fast"},
		{"medium", map[string]any{"p95_latency_ms": 5000}, "medium"},
		{"slow", map[string]any{"p95_latency_ms": 9000}, "slow"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			require.Equal(t, c.want, speedTierOf(c.snapshot))
		})
	}
}

func TestRawProvidedDistinguishesNull(t *testing.T) {
	provided, isNull := rawProvided(nil)
	require.False(t, provided)
	require.False(t, isNull)

	provided, isNull = rawProvided([]byte("null"))
	require.True(t, provided)
	require.True(t, isNull)

	provided, isNull = rawProvided([]byte(" 12 "))
	require.True(t, provided)
	require.False(t, isNull)
}

func TestApplyThresholdOverridesSeed(t *testing.T) {
	thresholds := seedThresholds
	applyThreshold(&thresholds, "error_rate_open", "0.5")
	applyThreshold(&thresholds, "min_window_calls", "42")
	applyThreshold(&thresholds, "circuit_cooldown", "120")
	applyThreshold(&thresholds, "unknown_key", "x")
	require.Equal(t, 0.5, thresholds.ErrorRateOpen)
	require.Equal(t, 42, thresholds.MinWindowCalls)
	require.Equal(t, 120, thresholds.CircuitCooldownSec)
	require.Equal(t, seedThresholds.ErrorRateWarn, thresholds.ErrorRateWarn)
}

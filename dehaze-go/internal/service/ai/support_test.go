package ai

import (
	"path/filepath"
	"testing"

	"github.com/joho/godotenv"
	"github.com/stretchr/testify/require"
)

// Python 侧（dehaze-python ai_cipher）用同一环境密钥与固定 IV(0..15) 生成的密文，
// 作为跨端互认基线：Go 必须能解出明文，密文格式（base64(iv+ct)，PKCS7）不得偏离。
const (
	pythonPlaintext  = "sk-test-1234567890abcdef"
	pythonCiphertext = "AAECAwQFBgcICQoLDA0OD04BvaVwbwVz3nUguEEnoh3yEJ+0wte2FqIoQ8LCAFp4"
	pythonKeyHash    = "93faf5716142f90256b4c1c4bc2c1bc2ee0a2460d5a738d6938dda1ad4216afd"
	pythonMaskedKey  = "sk-test-..."
)

func loadRepoEnv(t *testing.T) {
	t.Helper()
	// 测试 CWD 是包目录（dehaze-go/internal/service/ai），仓库根 .env 在上溯四级
	require.NoError(t, godotenv.Load(filepath.Join("..", "..", "..", "..", ".env")))
}

func TestDecryptSecretReadsPythonCiphertext(t *testing.T) {
	loadRepoEnv(t)
	plain, err := DecryptSecret(pythonCiphertext)
	require.NoError(t, err)
	require.Equal(t, pythonPlaintext, plain, "Go 必须能解开 python 写入的密文（跨端互认）")
}

func TestEncryptSecretRoundTrip(t *testing.T) {
	loadRepoEnv(t)
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

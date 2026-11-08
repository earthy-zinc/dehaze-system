package aicontract

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// 中断过期判定的时区口径：无时区的紧凑格式由三端按本地时间写入，必须按本地时区解析；
// 带 Z/偏移的格式按字面时区解析。两者表示同一时刻时必须等价。

// withLocalZone 临时改写进程本地时区，供解析用例在 UTC 容器里也能验证本地语义。
func withLocalZone(t *testing.T, loc *time.Location) {
	t.Helper()
	original := time.Local
	time.Local = loc
	t.Cleanup(func() { time.Local = original })
}

// TestParseInterruptTimeCompactLayoutsUseLocalZone 无时区串按本地时间解析，
// 且与带 +08:00 的同一时刻等价。
func TestParseInterruptTimeCompactLayoutsUseLocalZone(t *testing.T) {
	shanghai := time.FixedZone("CST", 8*3600)
	withLocalZone(t, shanghai)

	expected := time.Date(2026, 10, 21, 10, 0, 0, 0, shanghai)
	zoned, ok := parseInterruptTime("2026-10-21T10:00:00+08:00")
	require.True(t, ok, "带偏移的 RFC3339 必须可解析")

	for _, value := range []string{"2026-10-21T10:00:00", "2026-10-21 10:00:00"} {
		parsed, ok := parseInterruptTime(value)
		require.True(t, ok, "无时区串 %q 必须可解析", value)
		_, offset := parsed.Zone()
		require.Equal(t, 8*3600, offset, "无时区串 %q 必须按本地时区（+08:00）解析", value)
	}

	compact, ok := parseInterruptTime("2026-10-21T10:00:00")
	require.True(t, ok)
	require.True(t, compact.Equal(expected), "无时区串应等于本地 10:00")
	require.True(t, compact.Equal(zoned), "无时区串应与同一时刻的 +08:00 串等价")
}

// TestParseInterruptTimeZonedLayoutsKeepLiteralZone 带 Z 的串保持 UTC，不被本地时区改写。
func TestParseInterruptTimeZonedLayoutsKeepLiteralZone(t *testing.T) {
	withLocalZone(t, time.FixedZone("CST", 8*3600))

	parsed, ok := parseInterruptTime("2026-10-21T02:00:00Z")
	require.True(t, ok)
	_, offset := parsed.Zone()
	require.Equal(t, 0, offset, "带 Z 的串必须按 UTC 解析")
	require.True(t, parsed.Equal(time.Date(2026, 10, 21, 2, 0, 0, 0, time.UTC)))
}

// TestNormalizeInterruptExpiryUsesLocalZone 过期判定必须使用本地时区解析的 expiresAt：
// 若按 UTC 解析，本地 10:01 仍会被误判为未过期（差 8 小时）。
func TestNormalizeInterruptExpiryUsesLocalZone(t *testing.T) {
	shanghai := time.FixedZone("CST", 8*3600)
	withLocalZone(t, shanghai)

	interrupt := &Interrupt{ID: "it-1", Type: InterruptTypeConfirm, ExpiresAt: "2026-10-21T10:00:00", Actionable: true}

	before := NormalizeInterrupt(interrupt, time.Date(2026, 10, 21, 9, 59, 0, 0, shanghai))
	require.True(t, before.Actionable, "本地 09:59 未过期，可续")

	after := NormalizeInterrupt(interrupt, time.Date(2026, 10, 21, 10, 1, 0, 0, shanghai))
	require.False(t, after.Actionable, "本地 10:01 已过期，actionable 必须为 false")
	require.Equal(t, "it-1", after.ID, "payload 保留")
	require.Equal(t, "2026-10-21T10:00:00", after.ExpiresAt, "payload 保留")
}

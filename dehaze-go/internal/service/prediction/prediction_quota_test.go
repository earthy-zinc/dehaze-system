package prediction

import (
	"context"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/stretchr/testify/require"
)

// TestGetQuota_IncludesResetDate 配额结构必须含 resetDate（python `QuotaResponse.resetDate`，
// 格式 yyyy-MM-dd，月度配额为下月 1 日）。SDK model.test.ts 的"配额数据结构完整性"用例即断言该字段。
func TestGetQuota_IncludesResetDate(t *testing.T) {
	db := testutil.NewTestDB(t)
	ctx := context.Background()
	svc := newPredictionService(t, db, "http://127.0.0.1:1", newRealMemberService(t, db))

	userID := int64(881001)
	createMember(t, db, userID, 3) // 配额 5 / 已用 3

	quota, err := svc.GetQuota(ctx, userID)
	require.NoError(t, err)
	require.Equal(t, 5, quota.Total)
	require.Equal(t, 3, quota.Used)
	require.Equal(t, 2, quota.Remaining)

	now := time.Now()
	want := time.Date(now.Year(), now.Month()+1, 1, 0, 0, 0, 0, now.Location()).Format("2006-01-02")
	require.Equal(t, want, quota.ResetDate, "resetDate 必须为下月 1 日（python 口径）")
}

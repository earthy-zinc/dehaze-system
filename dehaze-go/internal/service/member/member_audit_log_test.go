package member_test

import (
	"context"
	"testing"
	"time"

	memberrepo "github.com/earthyzinc/dehaze-go/internal/repository/member"
	"github.com/earthyzinc/dehaze-go/internal/service/member"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/lifecycle"
	"github.com/stretchr/testify/require"
)

// stubAuditLister 手写桩：记录被查询的 target_type/target_id 并回放固定条目
type stubAuditLister struct {
	gotTargetType string
	gotTargetID   int64
	gotPage       int
	gotPageSize   int
	items         []member.MemberAuditLog
	total         int64
}

func (s *stubAuditLister) ListByTarget(
	_ context.Context, targetType string, targetID int64, page, pageSize int,
) ([]member.MemberAuditLog, int64, error) {
	s.gotTargetType, s.gotTargetID, s.gotPage, s.gotPageSize = targetType, targetID, page, pageSize
	return s.items, s.total, nil
}

// TestListMemberAuditLogs 会员操作日志口径（对齐 python `list_member_audit_logs`）：
// target_type 固定 "member"、target_id 为被查会员，只回 {list,total}，条目字段 camelCase 化，
// createTime 由 Mongo 的 UTC 转本地展示（python `_format_utc_dt` 同口径）。
func TestListMemberAuditLogs(t *testing.T) {
	db := testutil.NewTestDB(t)
	ctx := context.Background()
	utc := time.Date(2026, 9, 18, 3, 4, 5, 0, time.UTC)

	lister := &stubAuditLister{
		items: []member.MemberAuditLog{{
			ID: "68c9f0e1a1b2c3d4e5f60718", OperatorID: 2, Action: "level_change",
			Module: "member", BeforeValue: "level_0", AfterValue: "level_1",
			IP: "127.0.0.1", CreateTime: utc,
		}},
		total: 1,
	}

	lm := lifecycle.NewManager()
	t.Cleanup(func() { _ = lm.Shutdown(2 * time.Second) })
	svc := member.NewMemberService(
		db,
		memberrepo.NewMemberRepository(db),
		memberrepo.NewMemberBenefitRepository(db),
		memberrepo.NewMemberGrowthLogRepository(db),
		memberrepo.NewMemberSignInRepository(db),
		nil, nil, nil, lm, nil, nil,
		lister,
		nil, // aiCredits
		nil, // cardOverrides
	)

	page, err := svc.ListMemberAuditLogs(ctx, 4321, 2, 10)
	require.NoError(t, err)
	require.Equal(t, "member", lister.gotTargetType, "target_type 固定为 member")
	require.EqualValues(t, 4321, lister.gotTargetID)
	require.Equal(t, 2, lister.gotPage)
	require.Equal(t, 10, lister.gotPageSize)
	require.EqualValues(t, 1, page.Total)
	require.Len(t, page.List, 1)

	item := page.List[0]
	require.Equal(t, "68c9f0e1a1b2c3d4e5f60718", item.ID)
	require.EqualValues(t, 2, item.OperatorID)
	require.Equal(t, "level_change", item.Action)
	require.Equal(t, "level_0", item.BeforeValue)
	require.Equal(t, "level_1", item.AfterValue)
	require.Equal(t, utc.Local().Format("2006-01-02 15:04:05"), item.CreateTime)
}

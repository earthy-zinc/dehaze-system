package member_test

import (
	"context"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	memberrepo "github.com/earthyzinc/dehaze-go/internal/repository/member"
	"github.com/earthyzinc/dehaze-go/internal/service/member"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/lifecycle"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// stubTrialDeps 手写桩：只实现 trial-status 需要的三项取数（避免为只读端点引 mock 框架）
type stubTrialDeps struct {
	couponExpire *time.Time
	trialCredits int64
	paidOrder    bool
}

func (s stubTrialDeps) ActiveTrialCouponExpireTime(context.Context, int64) (*time.Time, error) {
	return s.couponExpire, nil
}

func (s stubTrialDeps) SumTrialCredits(context.Context, int64) (int64, error) {
	return s.trialCredits, nil
}

func (s stubTrialDeps) HasPaidOrder(context.Context, int64) (bool, error) {
	return s.paidOrder, nil
}

func newTrialService(t *testing.T, db *gorm.DB, deps member.TrialStatusDeps) *member.MemberService {
	t.Helper()
	lm := lifecycle.NewManager()
	t.Cleanup(func() { _ = lm.Shutdown(2 * time.Second) })
	return member.NewMemberService(
		db,
		memberrepo.NewMemberRepository(db),
		memberrepo.NewMemberBenefitRepository(db),
		memberrepo.NewMemberGrowthLogRepository(db),
		memberrepo.NewMemberSignInRepository(db),
		nil, // cache
		nil, // auditLogSvc
		nil, // messageSender
		lm,
		nil, // dictSvc
		deps,
		nil, // auditLister（操作日志用例见 member_audit_log_test.go）
		nil, // aiCredits
		nil, // cardOverrides
	)
}

// TestGetTrialStatus 试用引导状态口径（对齐 python `get_trial_status`）：
// voucherActivated＝持有有效 trial 券；showTrialEntry＝(未激活) 或 (试用积分>0) 或 (新用户专享可用)；
// newUserExclusiveAvailable＝无已支付订单；paidMembership＝等级来源为购买或存在到期时间。
func TestGetTrialStatus(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)
	ctx := context.Background()
	expire := time.Date(2026, 10, 1, 12, 30, 0, 0, time.Local)

	cases := []struct {
		name        string
		deps        stubTrialDeps
		member      *model.SysMember
		wantActive  bool
		wantShow    bool
		wantNewUser bool
		wantPaid    bool
		wantExpire  *string
	}{
		{
			name:     "新用户：无券/无积分/无订单 → 展示试用入口",
			deps:     stubTrialDeps{paidOrder: false},
			member:   newQuotaMember(991001, 0, 0, 0, 0, 1),
			wantShow: true, wantNewUser: true,
		},
		{
			name:       "已激活券 + 有试用积分 + 已付费 → 仍展示（积分>0）",
			deps:       stubTrialDeps{couponExpire: &expire, trialCredits: 50, paidOrder: true},
			member:     newQuotaMember(991002, 0, 0, 0, 0, 1),
			wantActive: true, wantShow: true,
			wantExpire: strPtr("2026-10-01 12:30:00"),
		},
		{
			name:       "已激活券 + 无积分 + 已付费 → 三项皆假，不展示",
			deps:       stubTrialDeps{couponExpire: &expire, trialCredits: 0, paidOrder: true},
			member:     newQuotaMember(991003, 0, 0, 0, 0, 1),
			wantActive: true, wantShow: false,
			wantExpire: strPtr("2026-10-01 12:30:00"),
		},
		{
			name:       "购买来源会员 → paidMembership 为真",
			deps:       stubTrialDeps{couponExpire: &expire, paidOrder: true},
			member:     purchasedMember(991004),
			wantActive: true, wantPaid: true,
			wantExpire: strPtr("2026-10-01 12:30:00"),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mustCreateMember(t, db, tc.member)
			svc := newTrialService(t, db, tc.deps)

			status, err := svc.GetTrialStatus(ctx, tc.member.UserID)
			require.NoError(t, err)
			require.Equal(t, tc.wantActive, status.VoucherActivated)
			require.Equal(t, tc.wantShow, status.ShowTrialEntry)
			require.Equal(t, tc.wantNewUser, status.NewUserExclusiveAvailable)
			require.Equal(t, tc.wantPaid, status.PaidMembership)
			require.Equal(t, 3, status.TrialDays)
			require.Equal(t, 100, status.TrialCredits)
			require.Equal(t, tc.deps.trialCredits, status.AITrialCreditsBalance)
			if tc.wantExpire == nil {
				require.Nil(t, status.VoucherExpireTime)
			} else {
				require.NotNil(t, status.VoucherExpireTime)
				require.Equal(t, *tc.wantExpire, *status.VoucherExpireTime)
			}
		})
	}
}

func strPtr(s string) *string { return &s }

func purchasedMember(userID int64) *model.SysMember {
	m := newQuotaMember(userID, 0, 0, 0, 0, 1)
	m.LevelSource = "purchase"
	return m
}

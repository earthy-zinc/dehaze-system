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

type stubAICredits struct{ balance, todayUsed int64 }

func (s stubAICredits) AICredits(context.Context, int64) (int64, int64, error) {
	return s.balance, s.todayUsed, nil
}

type stubOverrides struct{ overrides map[string]int }

func (s stubOverrides) PackageBenefitOverrides(context.Context, string) (map[string]int, error) {
	return s.overrides, nil
}

func newSummaryService(
	t *testing.T, db *gorm.DB, credits member.AICreditsProvider, overrides member.PackageOverridesProvider,
) *member.MemberService {
	t.Helper()
	lm := lifecycle.NewManager()
	t.Cleanup(func() { _ = lm.Shutdown(2 * time.Second) })
	return member.NewMemberService(
		db,
		memberrepo.NewMemberRepository(db),
		memberrepo.NewMemberBenefitRepository(db),
		memberrepo.NewMemberGrowthLogRepository(db),
		memberrepo.NewMemberSignInRepository(db),
		nil, nil, nil, lm, nil,
		nil, // trialDeps
		nil, // auditLister
		credits,
		overrides,
	)
}

// TestGetBenefitSummaryAICategory AI 类目口径（对齐 python `get_benefit_summary`）：
// 余额/今日已用取自 AI 计费模块；限额取"等级权益 vs 已购会员卡覆盖值"的**较高值**，
// 且覆盖项仅在"购买来源且未过期"的会员卡生效（python `resolve_card_overrides`）。
func TestGetBenefitSummaryAICategory(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)
	ctx := context.Background()

	// 自定义等级权益：日 100 / 月 500
	require.NoError(t, db.Create(&model.SysMemberBenefit{
		LevelCode: "test_ai_lv", LevelName: "AI 测试等级",
		AiCreditsDaily: 100, AiCreditsMonthly: 500, Status: 1,
	}).Error)

	notExpired := time.Now().Add(24 * time.Hour)

	newMember := func(userID int64, levelSource string, expire *time.Time) *model.SysMember {
		m := newQuotaMember(userID, 0, 0, 0, 0, 1)
		m.LevelCode = "test_ai_lv"
		m.LevelSource = levelSource
		m.ExpireTime = expire
		return m
	}

	t.Run("成长来源：限额取等级权益，余额/今日已用透传", func(t *testing.T) {
		mustCreateMember(t, db, newMember(992001, "growth", nil))
		svc := newSummaryService(t, db, stubAICredits{balance: 42, todayUsed: 7},
			stubOverrides{overrides: map[string]int{"aiCreditsDaily": 300, "aiCreditsMonthly": 900}})

		summary, err := svc.GetBenefitSummary(ctx, 992001)
		require.NoError(t, err)
		require.Equal(t, 42, summary.AICategory.CreditsBalance)
		require.Equal(t, 7, summary.AICategory.TodayUsed)
		require.Equal(t, 100, summary.AICategory.DailyLimit, "非购买来源不得采用会员卡覆盖值")
		require.Equal(t, 500, summary.AICategory.MonthlyLimit)
	})

	t.Run("购买来源未过期：取覆盖值与等级权益的较高值", func(t *testing.T) {
		mustCreateMember(t, db, newMember(992002, "purchase", &notExpired))
		svc := newSummaryService(t, db, stubAICredits{balance: 0, todayUsed: 0},
			stubOverrides{overrides: map[string]int{"aiCreditsDaily": 300, "aiCreditsMonthly": 200}})

		summary, err := svc.GetBenefitSummary(ctx, 992002)
		require.NoError(t, err)
		require.Equal(t, 300, summary.AICategory.DailyLimit, "覆盖值 300 > 权益 100")
		require.Equal(t, 500, summary.AICategory.MonthlyLimit, "覆盖值 200 < 权益 500，取较高值")
	})

	t.Run("购买来源已过期：回退等级权益", func(t *testing.T) {
		expired := time.Now().Add(-time.Hour)
		mustCreateMember(t, db, newMember(992003, "purchase", &expired))
		svc := newSummaryService(t, db, stubAICredits{balance: 0, todayUsed: 0},
			stubOverrides{overrides: map[string]int{"aiCreditsDaily": 300}})

		summary, err := svc.GetBenefitSummary(ctx, 992003)
		require.NoError(t, err)
		require.Equal(t, 100, summary.AICategory.DailyLimit, "会员卡已过期则回退等级权益")
	})
}

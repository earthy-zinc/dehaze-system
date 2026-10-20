// 会员模块测试公共 helper：真实 MySQL 测试库装配与会员行构造。
package member_test

import (
	"testing"
	"time"

	"github.com/earthyzinc/clarity-go/internal/model"
	memberrepo "github.com/earthyzinc/clarity-go/internal/repository/member"
	"github.com/earthyzinc/clarity-go/internal/service/member"
	"github.com/earthyzinc/clarity-go/pkg/cache/types"
	"github.com/earthyzinc/clarity-go/pkg/lifecycle"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// resetMemberTables 清空会员表，规避 config/sql 种子会员（user_id 4~8）对断言的干扰。
// 测试在独立事务内执行，回滚后种子数据自动恢复。
func resetMemberTables(t *testing.T, db *gorm.DB) {
	t.Helper()
	require.NoError(t, db.Exec("DELETE FROM sys_member").Error)
}

// newQuotaMember 构造会员行：默认 level_0/正常状态/未删除，配额字段显式注入。
func newQuotaMember(userID int64, dehazeQuota, dehazeUsed, evaluateQuota, evaluateUsed int, status int8) *model.SysMember {
	return &model.SysMember{
		UserID:               userID,
		LevelCode:            "level_0",
		LevelSource:          "growth",
		GrowthValue:          0,
		TotalConsumption:     0,
		Status:               status,
		MonthlyDehazeQuota:   dehazeQuota,
		MonthlyDehazeUsed:    dehazeUsed,
		MonthlyEvaluateQuota: evaluateQuota,
		MonthlyEvaluateUsed:  evaluateUsed,
		Deleted:              0,
	}
}

func mustCreateMember(t *testing.T, db *gorm.DB, m *model.SysMember) {
	t.Helper()
	require.NoError(t, db.Create(m).Error)
}

// newQuotaService 按实际构造注入组装：repository 用真实 MySQL 测试库，
// cache 用 MockICache 或 nil，audit/message 依赖按需传 nil。
func newQuotaService(t *testing.T, db *gorm.DB, cache types.ICache) *member.MemberService {
	t.Helper()
	lm := lifecycle.NewManager()
	t.Cleanup(func() { _ = lm.Shutdown(2 * time.Second) })
	return member.NewMemberService(
		db,
		memberrepo.NewMemberRepository(db),
		memberrepo.NewMemberBenefitRepository(db),
		memberrepo.NewMemberGrowthLogRepository(db),
		memberrepo.NewMemberSignInRepository(db),
		cache,
		nil, // auditLogSvc
		nil, // messageSender
		lm,
		nil, // dictSvc
		nil, // trialDeps
		nil, // auditLister
		nil, // aiCredits
		nil, // cardOverrides
	)
}

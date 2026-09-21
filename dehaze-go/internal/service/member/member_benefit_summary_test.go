package member_test

import (
	"context"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/stretchr/testify/require"
)

// TestGetBenefitSummary 权益概览取数契约（对齐 python `get_benefit_summary`）：
//   - 缺会员行时**自动初始化**（python `get_or_init_member`；detail 走 MEMBER_NOT_FOUND，两者不可混用）；
//   - imageCategory.remaining = 7 类图像任务剩余的最低值，details[] 顺序固定；
//   - evaluateCategory.remaining = monthly_evaluate_quota - monthly_evaluate_used。
func TestGetBenefitSummary(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)
	svc := newQuotaService(t, db, nil)
	ctx := context.Background()

	// 1) 缺会员行：必须自动初始化而非报错
	summary, err := svc.GetBenefitSummary(ctx, 990001)
	require.NoError(t, err, "缺会员行时 summary 走自动初始化，不得抛错")
	require.Len(t, summary.ImageCategory.Details, 7, "图像类目固定 7 类任务")

	taskTypes := make([]string, 0, len(summary.ImageCategory.Details))
	for _, d := range summary.ImageCategory.Details {
		taskTypes = append(taskTypes, d.TaskType)
	}
	require.Equal(t,
		[]string{"dehaze", "derain", "desnow", "lowlight", "super_resolution", "denoise", "inpaint"},
		taskTypes, "details 顺序必须与 python IMAGE_TASK_TYPES 一致")

	// 2) 取数：类别剩余取最低值；evaluate 取差值
	member := newQuotaMember(990002, 10, 3, 20, 6, 1)
	member.MonthlyDerainQuota, member.MonthlyDerainUsed = 5, 5 // remaining 0 → 成为最低值
	member.MonthlyInpaintQuota, member.MonthlyInpaintUsed = 8, 1
	mustCreateMember(t, db, member)

	summary, err = svc.GetBenefitSummary(ctx, 990002)
	require.NoError(t, err)
	require.Equal(t, 0, summary.ImageCategory.Remaining, "imageCategory.remaining 取各任务剩余最低值")

	byType := map[string]int{}
	for _, d := range summary.ImageCategory.Details {
		byType[d.TaskType] = d.Remaining
	}
	require.Equal(t, 7, byType["dehaze"])
	require.Equal(t, 7, byType["inpaint"])
	require.Equal(t, 14, summary.EvaluateCategory.Remaining)
}

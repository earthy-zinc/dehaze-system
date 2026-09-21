package eval_log

import (
	"context"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

func seedEvalLog(t *testing.T, db *gorm.DB, userID int64, status model.LogStatus, taskType string) *model.SysEvalLog {
	t.Helper()
	log := &model.SysEvalLog{
		BaseModel:   model.BaseModel{CreateBy: userID},
		AlgorithmID: 13,
		Status:      status,
		TaskType:    taskType,
	}
	require.NoError(t, db.Create(log).Error)
	return log
}

// TestFindLogPageByUser_IsolationAndAllStatuses 评估日志列表契约（对齐 python `list_logs` →
// `eval_log_repository.get_paginated`：**仅 `create_by = user_id`**，不按状态、不按 task_type 过滤，按 id 倒序）。
//
// 与"指标历史"（`FindPageByUser`：completed + task_type='evaluation'）刻意区分：
// 若把指标历史的口径误用到日志列表，用户自己"处理中/失败"的日志会被隐藏（SDK model.test.ts:471 即此现象）。
func TestFindLogPageByUser_IsolationAndAllStatuses(t *testing.T) {
	db := testutil.NewTestDB(t)
	repo := NewEvalLogRepository(db)
	ctx := context.Background()

	userA, userB := int64(870001), int64(870002)
	seedEvalLog(t, db, userA, model.LogStatusCompleted, "evaluation")
	seedEvalLog(t, db, userA, model.LogStatusFailed, "evaluation")
	seedEvalLog(t, db, userA, model.LogStatusProcessing, "evaluation")
	last := seedEvalLog(t, db, userA, model.LogStatusCompleted, "report")
	otherUserLog := seedEvalLog(t, db, userB, model.LogStatusCompleted, "evaluation")

	list, total, err := repo.FindLogPageByUser(ctx, userA, 0, 1, 100)
	require.NoError(t, err)
	require.EqualValues(t, 4, total, "日志列表必须包含各状态与 report 类型（python 不过滤）")
	require.Len(t, list, 4)
	require.Equal(t, last.ID, list[0].ID, "按 id 倒序（python desc(SysEvalLog.id)）")

	for _, item := range list {
		require.Equal(t, userA, item.CreateBy, "跨用户日志不得出现（按 create_by 隔离）")
		require.NotEqual(t, otherUserLog.ID, item.ID)
	}

	// 算法筛选与列表口径叠加
	filtered, filteredTotal, err := repo.FindLogPageByUser(ctx, userA, 13, 1, 100)
	require.NoError(t, err)
	require.EqualValues(t, 4, filteredTotal)
	require.Len(t, filtered, 4)

	// 无匹配算法时为空
	_, noneTotal, err := repo.FindLogPageByUser(ctx, userA, 9999, 1, 100)
	require.NoError(t, err)
	require.EqualValues(t, 0, noneTotal)
}

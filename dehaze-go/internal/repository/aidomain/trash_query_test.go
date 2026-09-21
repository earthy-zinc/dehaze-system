package aidomain

import (
	"context"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/database"
	"github.com/stretchr/testify/require"
)

// TestTrashQueriesBypassSoftDeleteCallback 回归（SDK go 集成 `:344` 会话恢复 / `:586` 记忆恢复）：
// 生产库初始化时注册了全局软删回调（pkg/database/soft_delete.go：对所有含 Deleted 字段的模型追加
// `deleted = 0`），而回收站/恢复类查询要的是 `deleted <> 0` —— 不加 Unscoped 就永远查不到软删行
// （生产 SQL 实证：`... AND deleted <> 0 ... AND sys_ai_conversation.deleted = 0`，rows=0）。
//
// 本用例**刻意注册同一回调**，否则测试库（未注册）会让用例恒绿，失去保护意义。
func TestTrashQueriesBypassSoftDeleteCallback(t *testing.T) {
	db := database.RegisterSoftDeleteCallback(testutil.NewTestDB(t))
	ctx := context.Background()
	windowStart := time.Now().AddDate(0, 0, -30)

	t.Run("会话：软删后 GetInTrash/PaginateTrash 必须命中，恢复后回到可见集", func(t *testing.T) {
		repo := NewConversationRepository(db)
		conv := &model.SysAiConversation{
			UserID: 990001, Title: "回收站查询回归", TitleSource: "auto", Status: 1,
		}
		require.NoError(t, repo.Create(ctx, conv))
		require.NoError(t, repo.SoftDeleteByIDs(ctx, []int64{conv.ID}, 990001))

		found, err := repo.GetInTrash(ctx, conv.ID, 990001, windowStart)
		require.NoError(t, err)
		require.NotNil(t, found, "软删会话必须能被恢复查询命中（否则恢复必报已超出恢复窗口）")

		items, total, err := repo.PaginateTrash(ctx, 990001, 1, 10, windowStart)
		require.NoError(t, err)
		require.EqualValues(t, 1, total)
		require.Len(t, items, 1)

		require.NoError(t, repo.RestoreByIDs(ctx, []int64{conv.ID}, 990001))
		active, err := repo.GetByID(ctx, conv.ID)
		require.NoError(t, err)
		require.NotNil(t, active, "恢复后必须回到常规（deleted = 0）可见集")
	})

	t.Run("记忆：恢复窗口内软删行必须可查且可恢复", func(t *testing.T) {
		repo := NewMemoryRepository(db)
		memory := &model.SysAiMemory{
			UserID: 990001, MemoryType: "semantic", Content: "恢复窗口回归",
			Source: "manual", Status: 1,
		}
		require.NoError(t, repo.Create(ctx, memory))

		cleared, err := repo.BatchClear(ctx, 990001, "semantic", nil, nil)
		require.NoError(t, err)
		require.GreaterOrEqual(t, cleared, int64(1))

		deleted, err := repo.ListDeletedForRestore(ctx, 990001, "semantic", nil, nil)
		require.NoError(t, err)
		require.NotEmpty(t, deleted, "恢复窗口内的软删记忆必须可查（否则恢复数恒为 0）")

		restored, err := repo.RestoreDeleted(ctx, []int64{memory.ID})
		require.NoError(t, err)
		require.EqualValues(t, 1, restored)
	})
}

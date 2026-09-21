package database

import (
	"context"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/stretchr/testify/require"
)

// TestAutoFillUpdateBySkipsTablesWithoutColumn 回归（SDK go 集成 P0①：Agent 回滚 `POST /ai/agents/{id}/versions/{n}/rollback`
// 报 B0001「更新版本状态失败」）：`autoFillUpdateBy` 的 Map 分支会把 `update_by` 无条件并入 SET 子句，
// 而 `sys_ai_agent_version` 等表本就没有该列 → `Error 1054 Unknown column 'update_by'` → 更新整体失败。
// 生产 `DemotePublished` 正是 `Model(&SysAiAgentVersion{}).Where(...).Update("status", 1)`（Dest 为 map，故走该分支）。
func TestAutoFillUpdateBySkipsTablesWithoutColumn(t *testing.T) {
	db := RegisterGormCallbacks(testutil.NewTestDB(t))
	ctx := SetUserID(context.Background(), 999001)

	t.Run("无 update_by 列的表：更新必须成功", func(t *testing.T) {
		version := &model.SysAiAgentVersion{AgentID: 990001, VersionNo: 1, Snapshot: "{}", Status: 2}
		require.NoError(t, db.Create(version).Error)

		err := db.WithContext(ctx).Model(&model.SysAiAgentVersion{}).
			Where("agent_id = ? AND status = 2", 990001).Update("status", 1).Error
		require.NoError(t, err, "表无 update_by 列时不得注入 update_by（否则 1054）")

		var status int
		require.NoError(t, db.Raw("SELECT status FROM sys_ai_agent_version WHERE id = ?", version.ID).
			Row().Scan(&status))
		require.Equal(t, 1, status, "状态必须真的被更新")
	})

	t.Run("有 update_by 列的表：仍自动填充", func(t *testing.T) {
		agent := &model.SysAiAgent{
			AgentCode: "callback-probe", Name: "回调回归", ModelID: "m", ReasoningMode: "auto", Status: 1,
		}
		require.NoError(t, db.Create(agent).Error)

		require.NoError(t, db.WithContext(ctx).Model(&model.SysAiAgent{}).
			Where("id = ?", agent.ID).Update("name", "回调回归-改名").Error)

		var updateBy int64
		require.NoError(t, db.Raw("SELECT update_by FROM sys_ai_agent WHERE id = ?", agent.ID).
			Row().Scan(&updateBy))
		require.EqualValues(t, 999001, updateBy, "有该列时必须继续自动填充，不能因修复而失效")
	})
}

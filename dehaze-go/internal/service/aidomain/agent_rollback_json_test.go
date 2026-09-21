package aidomain

import (
	"context"
	"database/sql"
	"encoding/json"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	repo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/stretchr/testify/require"
)

// TestRollbackWritesNullForEmptyJsonColumns 回归（SDK go 集成 `POST /ai/agents/{id}/versions/{n}/rollback`
// 报 3140）：回滚走 **map 型更新**（`Updates(map)` 不跳过零值，model 的 `default:null` 对它无效），
// 快照里 `config`/`permissions` 为空时必须写 SQL NULL（对齐 python `snapshot.get("permissions")` 的
// None→NULL）。此前经 `marshalJSON(nil)` 得到空串 → MySQL 报
// `Invalid JSON text: "The document is empty." ... for column 'sys_ai_agent.permissions'`。
func TestRollbackWritesNullForEmptyJsonColumns(t *testing.T) {
	db := testutil.NewTestDB(t)
	ctx := context.Background()
	agents := repo.NewAgentRepository(db)
	svc := NewAgentVersionService(agents, nil)

	agent := &model.SysAiAgent{
		AgentCode: "rollback-json-probe", Name: "回滚空JSON回归",
		ModelID: "gpt-4o-mini", ReasoningMode: "auto", Status: 1,
	}
	require.NoError(t, agents.Create(ctx, agent))

	operatorID := int64(1)
	version, err := svc.writeVersion(ctx, agent, &operatorID, "初始发布", 2)
	require.NoError(t, err)
	require.NotZero(t, version.VersionNo)

	_, err = svc.Rollback(ctx, agent.ID, version.VersionNo, operatorID)
	require.NoError(t, err, "空 config/permissions 回滚必须写 NULL，而非空串（3140）")

	var permissions, config sql.NullString
	require.NoError(t, db.Raw("SELECT permissions, config FROM sys_ai_agent WHERE id = ?", agent.ID).
		Row().Scan(&permissions, &config))
	require.False(t, permissions.Valid, "空 permissions 必须落 NULL")
	// 快照中的 config 会带上 resolved_config 默认值（非空），回滚应写回合法 JSON 文本
	require.True(t, config.Valid, "resolved config 必须写回", config.String)
	require.True(t, json.Valid([]byte(config.String)), "config 必须是合法 JSON: %s", config.String)
}

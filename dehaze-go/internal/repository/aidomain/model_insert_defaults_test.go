package aidomain

import (
	"context"
	"database/sql"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/stretchr/testify/require"
)

// TestCreateFillsTimeAndJsonDefaults 回归（SDK go 集成 1307 passed/139 failed 的根因）：
// AI 域 model 此前 CreateTime 无 autoCreateTime、JSON 字符串列无 default，插入时直接下发
// time.Time 零值（'0000-00-00 00:00:00'）与 JSON 空串（”），在
// STRICT_TRANS_TABLES + NO_ZERO_DATE 的 MySQL 下分别报 1292 / 3140 → 创建会话/Agent/记忆全 500。
// 现由 model 标签兜底：create_time/update_time 自动回填；JSON 字符串列零值写 NULL；
// 唯一 NOT NULL 且无数据库默认值的 JSON 列 `sys_ai_agent_version.snapshot` 兜底写 '{}'。
func TestCreateFillsTimeAndJsonDefaults(t *testing.T) {
	db := testutil.NewTestDB(t)
	ctx := context.Background()

	t.Run("会话：时间自动回填 + model_config 空串写 NULL", func(t *testing.T) {
		conv := &model.SysAiConversation{
			UserID: 990001, Title: "插入默认值回归", TitleSource: "auto", Status: 1,
		}
		require.NoError(t, NewConversationRepository(db).Create(ctx, conv),
			"零值 create_time 不得直接下发（严格模式会报 1292）")
		require.False(t, conv.CreateTime.IsZero(), "autoCreateTime 必须回填 createTime")
		require.NotNil(t, conv.UpdateTime, "autoUpdateTime 必须回填 updateTime")

		var modelConfig sql.NullString
		require.NoError(t, db.Raw("SELECT model_config FROM sys_ai_conversation WHERE id = ?", conv.ID).
			Row().Scan(&modelConfig))
		require.False(t, modelConfig.Valid, "空 model_config 必须落 NULL，而非空串（3140）")
	})

	t.Run("Agent：config/permissions/tags 空串写 NULL", func(t *testing.T) {
		agent := &model.SysAiAgent{
			AgentCode: "insert-defaults-probe", Name: "插入默认值回归",
			ModelID: "m-probe", ReasoningMode: "auto", Status: 1,
		}
		require.NoError(t, NewAgentRepository(db).Create(ctx, agent))
		require.False(t, agent.CreateTime.IsZero())
		require.NotNil(t, agent.UpdateTime)

		var cfg, perm, tags sql.NullString
		require.NoError(t, db.Raw(
			"SELECT config, permissions, tags FROM sys_ai_agent WHERE id = ?", agent.ID).
			Row().Scan(&cfg, &perm, &tags))
		require.False(t, cfg.Valid, "空 config 必须落 NULL")
		require.False(t, perm.Valid, "空 permissions 必须落 NULL")
		require.False(t, tags.Valid, "空 tags 必须落 NULL")
	})

	t.Run("记忆：metadata 空串写 NULL", func(t *testing.T) {
		memory := &model.SysAiMemory{
			UserID: 990001, MemoryType: "fact", Content: "插入默认值回归",
			Source: "manual", Status: 1,
		}
		require.NoError(t, NewMemoryRepository(db).Create(ctx, memory))
		require.False(t, memory.CreateTime.IsZero())
		require.NotNil(t, memory.UpdateTime)

		var metadata sql.NullString
		require.NoError(t, db.Raw("SELECT metadata FROM sys_ai_memory WHERE id = ?", memory.ID).
			Row().Scan(&metadata))
		require.False(t, metadata.Valid, "空 metadata 必须落 NULL")
	})

	t.Run("消息：metadata/tool_calls/used_memory_ids 空串写 NULL", func(t *testing.T) {
		conv := &model.SysAiConversation{
			UserID: 990001, Title: "消息归属会话", TitleSource: "auto", Status: 1,
		}
		require.NoError(t, NewConversationRepository(db).Create(ctx, conv))

		msg := &model.SysAiMessage{ConversationID: conv.ID, Role: "user", Status: 1}
		require.NoError(t, db.Create(msg).Error)
		require.False(t, msg.CreateTime.IsZero())

		var metadata, toolCalls, usedMemoryIDs sql.NullString
		require.NoError(t, db.Raw(
			"SELECT metadata, tool_calls, used_memory_ids FROM sys_ai_message WHERE id = ?", msg.ID).
			Row().Scan(&metadata, &toolCalls, &usedMemoryIDs))
		require.False(t, metadata.Valid)
		require.False(t, toolCalls.Valid)
		require.False(t, usedMemoryIDs.Valid)
	})

	t.Run("Agent 版本：NOT NULL 的 snapshot 空串兜底写 '{}'", func(t *testing.T) {
		agent := &model.SysAiAgent{
			AgentCode: "insert-defaults-version-probe", Name: "版本快照回归",
			ModelID: "m-probe", ReasoningMode: "auto", Status: 1,
		}
		require.NoError(t, NewAgentRepository(db).Create(ctx, agent))

		version := &model.SysAiAgentVersion{AgentID: agent.ID, VersionNo: 1, Status: 1}
		require.NoError(t, NewAgentRepository(db).CreateVersion(ctx, version))
		require.False(t, version.CreateTime.IsZero())

		var snapshot string
		require.NoError(t, db.Raw("SELECT snapshot FROM sys_ai_agent_version WHERE id = ?", version.ID).
			Row().Scan(&snapshot))
		require.Equal(t, "{}", snapshot, "NOT NULL 的 JSON 列不得为空串，须兜底合法 JSON")
	})
}

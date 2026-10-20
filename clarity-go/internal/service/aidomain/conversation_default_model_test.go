package aidomain

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/earthyzinc/clarity-go/internal/model"
	"github.com/earthyzinc/clarity-go/internal/model/bo"
	airepo "github.com/earthyzinc/clarity-go/internal/repository/ai"
	aidomainrepo "github.com/earthyzinc/clarity-go/internal/repository/aidomain"
	dictrepo "github.com/earthyzinc/clarity-go/internal/repository/dict"
	aiservice "github.com/earthyzinc/clarity-go/internal/service/ai"
	dictservice "github.com/earthyzinc/clarity-go/internal/service/dict"
	"github.com/earthyzinc/clarity-go/internal/testutil"
	"github.com/earthyzinc/clarity-go/pkg/common"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

var convProbeSeq int64

func convProbeID() int64 {
	convProbeSeq++
	return time.Now().UnixNano() + convProbeSeq
}

func newConversationFixture(t *testing.T) (*ConversationService, *gorm.DB, *aiservice.UserPreferenceService) {
	t.Helper()
	testutil.LoadTestConfig(t)
	db := testutil.NewTestDB(t)
	preferences := aiservice.NewUserPreferenceService(
		airepo.NewUserPreferenceRepository(db),
		airepo.NewModelRepository(db),
		dictservice.NewDictService(dictrepo.NewDictRepository(db), dictrepo.NewDictTypeRepository(db), nil),
		aiservice.NewHealthService(db),
		aidomainrepo.NewAgentRepository(db),
	)
	service := NewConversationService(
		aidomainrepo.NewConversationRepository(db),
		aidomainrepo.NewMessageRepository(db),
		aidomainrepo.NewAgentRepository(db),
		preferences,
	)
	return service, db, preferences
}

// TestConversationCreateResolvesDefaultModel 覆盖契约⑤⑥⑦：新建会话未指定模型时按
// 「用户偏好 → 平台默认」解析；平台默认未配置则报 A0500，不回落代码常量。
func TestConversationCreateResolvesDefaultModel(t *testing.T) {
	service, db, preferences := newConversationFixture(t)
	ctx := context.Background()

	// 未设偏好 → 平台默认模型（sys_dict: ai_model.default_model）
	platformDefault := fmt.Sprintf("conv_probe_platform_%d", convProbeID())
	require.NoError(t, db.Exec(
		"UPDATE sys_dict SET value = ? WHERE type_code = 'ai_model' AND name = 'default_model'",
		platformDefault).Error)

	plainUser := convProbeID()
	conv, err := service.Create(ctx, plainUser, &ConversationCreateForm{})
	require.NoError(t, err)
	require.Equal(t, platformDefault, conv.Model)

	// 设了偏好 → 取用户偏好
	prefUser := convProbeID()
	preferred := fmt.Sprintf("conv_pref_probe_%d", prefUser)
	require.NoError(t, db.Create(&model.SysAiModel{
		ProviderID: 1, ModelID: preferred, ModelType: "chat", DisplayName: preferred,
		MaxContextTokens: 4096, MaxOutputTokens: 2048, SupportsStreaming: 1, Status: 1,
	}).Error)
	require.NoError(t, preferences.Update(ctx, prefUser, &bo.UserPreferenceForm{DefaultModel: &preferred}))

	conv, err = service.Create(ctx, prefUser, &ConversationCreateForm{})
	require.NoError(t, err)
	require.Equal(t, preferred, conv.Model)

	// 请求显式指定优先于偏好
	explicit := "gpt-4o-mini"
	conv, err = service.Create(ctx, prefUser, &ConversationCreateForm{Model: &explicit})
	require.NoError(t, err)
	require.Equal(t, explicit, conv.Model)

	// 平台默认未配置 → A0500，且不落库
	require.NoError(t, db.Exec(
		"DELETE FROM sys_dict WHERE type_code = 'ai_model' AND name = 'default_model'").Error)
	lateUser := convProbeID()
	_, err = service.Create(ctx, lateUser, &ConversationCreateForm{})
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "应为业务错误，实际 err=%v", err)
	require.Equal(t, common.BUSINESS_ERROR.Code, bizErr.Code().Code)
	require.Contains(t, bizErr.Message(), "未配置平台默认模型")

	var created int64
	require.NoError(t, db.Table("sys_ai_conversation").Where("user_id = ?", lateUser).Count(&created).Error)
	require.Zero(t, created, "解析失败不得创建会话")
}

// TestConversationCreateResolvesAgentModel 覆盖 Agent 级解析：
// 用户偏好 > Agent 配置模型（已发布快照优先，无已发布版本回退主表）> 平台默认；
// 切换 Agent 且未显式指定模型时会话模型随之重解析。
func TestConversationCreateResolvesAgentModel(t *testing.T) {
	service, db, preferences := newConversationFixture(t)
	ctx := context.Background()

	platformDefault := fmt.Sprintf("agent_platform_%d", convProbeID())
	require.NoError(t, db.Exec(
		"DELETE FROM sys_dict WHERE type_code = 'ai_model' AND name = 'default_model'").Error)
	require.NoError(t, db.Exec(
		"INSERT INTO sys_dict (type_code, name, value, status, deleted) "+
			"VALUES ('ai_model', 'default_model', ?, 1, 0)", platformDefault).Error)

	// Agent：主表 model=main，已发布快照 model=published → 取快照（运行面口径）
	code := fmt.Sprintf("probe_agent_%d", convProbeID())
	agent := &model.SysAiAgent{
		AgentCode: code, Name: code, ModelID: "agent-main-model",
		ReasoningMode: "auto", Status: 1,
	}
	require.NoError(t, db.Create(agent).Error)
	require.NoError(t, db.Create(&model.SysAiAgentVersion{
		AgentID: agent.ID, VersionNo: 1,
		Snapshot: `{"model_id":"agent-published-model"}`, Status: 2,
	}).Error)

	conv, err := service.Create(ctx, convProbeID(), &ConversationCreateForm{AgentCode: &code})
	require.NoError(t, err)
	require.Equal(t, "agent-published-model", conv.Model)

	// 用户偏好优先于 Agent 模型
	prefUser := convProbeID()
	preferred := fmt.Sprintf("agent_pref_%d", prefUser)
	require.NoError(t, db.Create(&model.SysAiModel{
		ProviderID: 1, ModelID: preferred, ModelType: "chat", DisplayName: preferred,
		MaxContextTokens: 4096, MaxOutputTokens: 2048, SupportsStreaming: 1, Status: 1,
	}).Error)
	require.NoError(t, preferences.Update(ctx, prefUser, &bo.UserPreferenceForm{DefaultModel: &preferred}))
	conv, err = service.Create(ctx, prefUser, &ConversationCreateForm{AgentCode: &code})
	require.NoError(t, err)
	require.Equal(t, preferred, conv.Model)

	// 无已发布版本 → 回退主表可编辑态模型
	code2 := fmt.Sprintf("probe_agent2_%d", convProbeID())
	require.NoError(t, db.Create(&model.SysAiAgent{
		AgentCode: code2, Name: code2, ModelID: "agent-main-only",
		ReasoningMode: "auto", Status: 1,
	}).Error)
	conv, err = service.Create(ctx, convProbeID(), &ConversationCreateForm{AgentCode: &code2})
	require.NoError(t, err)
	require.Equal(t, "agent-main-only", conv.Model)

	// 未指定 Agent → 平台默认；再切换 Agent（未传 model）→ 模型随新 Agent 重解析
	owner := convProbeID()
	conv, err = service.Create(ctx, owner, &ConversationCreateForm{})
	require.NoError(t, err)
	require.Equal(t, platformDefault, conv.Model)

	updated, err := service.Update(ctx, conv.ID, owner, &ConversationUpdateForm{AgentCode: &code2})
	require.NoError(t, err)
	require.Equal(t, "agent-main-only", updated.Model)
}

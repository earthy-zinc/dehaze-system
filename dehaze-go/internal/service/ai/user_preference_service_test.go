package ai

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	aidomainrepo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	dictrepo "github.com/earthyzinc/dehaze-go/internal/repository/dict"
	dictservice "github.com/earthyzinc/dehaze-go/internal/service/dict"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

var probeSeq int64

// uniqueProbeID 生成进程内唯一标识：用户 id / model_id 均不得撞种子数据，否则用例互相污染。
func uniqueProbeID() int64 {
	probeSeq++
	return time.Now().UnixNano() + probeSeq
}

func newUserPreferenceFixture(t *testing.T) (*UserPreferenceService, *gorm.DB) {
	t.Helper()
	testutil.LoadTestConfig(t)
	db := testutil.NewTestDB(t)
	return NewUserPreferenceService(
		airepo.NewUserPreferenceRepository(db),
		airepo.NewModelRepository(db),
		dictservice.NewDictService(dictrepo.NewDictRepository(db), dictrepo.NewDictTypeRepository(db), nil),
		NewHealthService(db),
		aidomainrepo.NewAgentRepository(db),
	), db
}

// seedProbeModel 造一条模型行：status 是带 default 标签的零值字段，GORM Create 会跳过它让库默认值生效，
// 故显式 INSERT，保证 status=0（已停用）用例真实造出停用行。
func seedProbeModel(t *testing.T, db *gorm.DB, modelID, modelType string, status, vipLevel int8) {
	t.Helper()
	require.NoError(t, db.Exec(
		"INSERT INTO sys_ai_model (provider_id, model_id, model_type, display_name, max_context_tokens, max_output_tokens, supports_streaming, status, vip_level, deleted) "+
			"VALUES (1, ?, ?, ?, 4096, 2048, 1, ?, ?, 0)",
		modelID, modelType, modelID, status, vipLevel).Error)
}

func seedMemberLevel(t *testing.T, db *gorm.DB, userID int64, levelCode string) {
	t.Helper()
	require.NoError(t, db.Exec(
		"INSERT INTO sys_member (user_id, level_code, deleted) VALUES (?, ?, 0)", userID, levelCode).Error)
}

func requireModelUnavailable(t *testing.T, err error) {
	t.Helper()
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "应为业务错误，实际 err=%v", err)
	require.Equal(t, common.AI_MODEL_NOT_AVAILABLE.Code, bizErr.Code().Code)
	require.Equal(t, "模型不存在、未启用或当前等级不可用", bizErr.Message())
}

// TestUserPreferenceReadWrite 覆盖契约①②③：无记录读 null 且不建行、设置可读回、传 null/缺失即清除。
func TestUserPreferenceReadWrite(t *testing.T) {
	svc, db := newUserPreferenceFixture(t)
	ctx := context.Background()
	userID := uniqueProbeID()

	got, err := svc.Get(ctx, userID)
	require.NoError(t, err)
	require.Nil(t, got.DefaultModel)
	var rowCount int64
	require.NoError(t, db.Table("sys_ai_user_preference").Where("user_id = ?", userID).Count(&rowCount).Error)
	require.Zero(t, rowCount, "GET 不得自动建行")

	modelID := fmt.Sprintf("pref_probe_chat_%d", userID)
	seedProbeModel(t, db, modelID, "chat", 1, 0)
	require.NoError(t, svc.Update(ctx, userID, &bo.UserPreferenceForm{DefaultModel: &modelID}))

	got, err = svc.Get(ctx, userID)
	require.NoError(t, err)
	require.NotNil(t, got.DefaultModel)
	require.Equal(t, modelID, *got.DefaultModel)

	var pref model.SysAiUserPreference
	require.NoError(t, db.Where("user_id = ?", userID).First(&pref).Error)
	require.NotNil(t, pref.CreateBy)
	require.Equal(t, userID, *pref.CreateBy)
	require.Equal(t, userID, *pref.UpdateBy)

	// 字段缺失＝清除偏好（与显式 null 等价，表单为指针时二者不可区分）
	require.NoError(t, svc.Update(ctx, userID, &bo.UserPreferenceForm{}))
	got, err = svc.Get(ctx, userID)
	require.NoError(t, err)
	require.Nil(t, got.DefaultModel)
	require.NoError(t, db.Where("user_id = ?", userID).First(&pref).Error)
	require.Nil(t, pref.DefaultModel)
	require.Equal(t, userID, *pref.UpdateBy)
}

// TestUserPreferenceUpdateRejectsUnusableModel 覆盖契约④：不存在/已停用/非 chat/超会员等级 → A0601，且拒绝时不落库。
func TestUserPreferenceUpdateRejectsUnusableModel(t *testing.T) {
	svc, db := newUserPreferenceFixture(t)
	ctx := context.Background()
	userID := uniqueProbeID()

	missing := fmt.Sprintf("pref_probe_missing_%d", userID)
	requireModelUnavailable(t, svc.Update(ctx, userID, &bo.UserPreferenceForm{DefaultModel: &missing}))

	disabled := fmt.Sprintf("pref_probe_disabled_%d", userID)
	seedProbeModel(t, db, disabled, "chat", 0, 0)
	requireModelUnavailable(t, svc.Update(ctx, userID, &bo.UserPreferenceForm{DefaultModel: &disabled}))

	embedding := fmt.Sprintf("pref_probe_embedding_%d", userID)
	seedProbeModel(t, db, embedding, "embedding", 1, 0)
	requireModelUnavailable(t, svc.Update(ctx, userID, &bo.UserPreferenceForm{DefaultModel: &embedding}))

	vipModel := fmt.Sprintf("pref_probe_vip_%d", userID)
	seedProbeModel(t, db, vipModel, "chat", 1, 2)
	lowLevelUser := uniqueProbeID()
	seedMemberLevel(t, db, lowLevelUser, "level_0")
	requireModelUnavailable(t, svc.Update(ctx, lowLevelUser, &bo.UserPreferenceForm{DefaultModel: &vipModel}))

	var rowCount int64
	require.NoError(t, db.Table("sys_ai_user_preference").Where("user_id = ?", lowLevelUser).Count(&rowCount).Error)
	require.Zero(t, rowCount, "校验失败不得落库")

	// 同一模型对等级足够的用户放行（等级口径与 /ai/models/enabled 同源）
	highLevelUser := uniqueProbeID()
	seedMemberLevel(t, db, highLevelUser, "level_2")
	require.NoError(t, svc.Update(ctx, highLevelUser, &bo.UserPreferenceForm{DefaultModel: &vipModel}))
}

// TestResolveChatModelChain 覆盖契约⑤⑥：显式指定 > 用户偏好 > 平台默认；偏好失效静默回落。
func TestResolveChatModelChain(t *testing.T) {
	svc, db := newUserPreferenceFixture(t)
	ctx := context.Background()

	// 平台默认取自 sys_dict：改字典值后解析结果随之变化，证明没有代码常量兜底
	platformDefault := fmt.Sprintf("pref_probe_platform_%d", uniqueProbeID())
	require.NoError(t, db.Exec(
		"UPDATE sys_dict SET value = ? WHERE type_code = 'ai_model' AND name = 'default_model'",
		platformDefault).Error)
	resolved, err := svc.PlatformDefaultChatModel(ctx)
	require.NoError(t, err)
	require.Equal(t, platformDefault, resolved)

	userID := uniqueProbeID()
	preferred := fmt.Sprintf("pref_probe_chain_%d", userID)
	seedProbeModel(t, db, preferred, "chat", 1, 0)
	require.NoError(t, svc.Update(ctx, userID, &bo.UserPreferenceForm{DefaultModel: &preferred}))

	explicit := "gpt-4o-mini"
	resolved, err = svc.ResolveChatModel(ctx, userID, &explicit, "")
	require.NoError(t, err)
	require.Equal(t, explicit, resolved)

	resolved, err = svc.ResolveChatModel(ctx, userID, nil, "")
	require.NoError(t, err)
	require.Equal(t, preferred, resolved)

	empty := ""
	resolved, err = svc.ResolveChatModel(ctx, userID, &empty, "")
	require.NoError(t, err)
	require.Equal(t, preferred, resolved)

	// 偏好指向的模型被停用 → 静默跳过，回落平台默认（不报错）
	require.NoError(t, db.Model(&model.SysAiModel{}).Where("model_id = ?", preferred).Update("status", 0).Error)
	resolved, err = svc.ResolveChatModel(ctx, userID, nil, "")
	require.NoError(t, err)
	require.Equal(t, platformDefault, resolved)

	resolved, err = svc.ResolveChatModel(ctx, uniqueProbeID(), nil, "")
	require.NoError(t, err)
	require.Equal(t, platformDefault, resolved)
}

// TestResolveChatModelRequiresPlatformDefault 覆盖契约⑦：平台默认未配置 → A0500，不回退代码常量。
func TestResolveChatModelRequiresPlatformDefault(t *testing.T) {
	svc, db := newUserPreferenceFixture(t)
	ctx := context.Background()

	require.NoError(t, db.Exec(
		"DELETE FROM sys_dict WHERE type_code = 'ai_model' AND name = 'default_model'").Error)

	_, err := svc.ResolveChatModel(ctx, uniqueProbeID(), nil, "")
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "应为业务错误，实际 err=%v", err)
	require.Equal(t, common.BUSINESS_ERROR.Code, bizErr.Code().Code)
	require.Contains(t, bizErr.Message(), "未配置平台默认模型")

	explicit := "gpt-4o-mini"
	resolved, err := svc.ResolveChatModel(ctx, uniqueProbeID(), &explicit, "")
	require.NoError(t, err)
	require.Equal(t, explicit, resolved)
}

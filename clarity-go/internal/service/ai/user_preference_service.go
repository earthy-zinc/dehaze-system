package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/earthyzinc/clarity-go/internal/model"
	"github.com/earthyzinc/clarity-go/internal/model/bo"
	"github.com/earthyzinc/clarity-go/internal/model/vo"
	airepo "github.com/earthyzinc/clarity-go/internal/repository/ai"
	aidomainrepo "github.com/earthyzinc/clarity-go/internal/repository/aidomain"
	"github.com/earthyzinc/clarity-go/internal/service/dict"
	"github.com/earthyzinc/clarity-go/pkg/common"
)

// UserPreferenceService 用户偏好默认模型：读写偏好 + 对话默认模型解析链。
// 本服务只服务"面向用户"的对话默认模型；后台派生推理（标题生成/记忆抽取/建议生成/试运行）不读偏好，取 PlatformDefaultChatModel。
type UserPreferenceService struct {
	preferences *airepo.UserPreferenceRepository
	models      *airepo.ModelRepository
	dicts       dict.IDictService
	health      *HealthService
	agents      *aidomainrepo.AgentRepository
}

func NewUserPreferenceService(
	preferences *airepo.UserPreferenceRepository,
	models *airepo.ModelRepository,
	dicts dict.IDictService,
	health *HealthService,
	agents *aidomainrepo.AgentRepository,
) *UserPreferenceService {
	return &UserPreferenceService{
		preferences: preferences,
		models:      models,
		dicts:       dicts,
		health:      health,
		agents:      agents,
	}
}

// Get 读取用户偏好默认模型；无记录或未设置返回 defaultModel=null，不自动建行
func (s *UserPreferenceService) Get(ctx context.Context, userID int64) (*vo.UserPreferenceVO, error) {
	pref, err := s.preferences.FindByUserID(ctx, userID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询用户偏好失败", err)
	}
	result := &vo.UserPreferenceVO{}
	if pref != nil && pref.DefaultModel != nil && *pref.DefaultModel != "" {
		result.DefaultModel = pref.DefaultModel
	}
	return result, nil
}

// Update 设置或清除用户偏好默认模型（form.DefaultModel 为 nil 即清除）
func (s *UserPreferenceService) Update(ctx context.Context, userID int64, form *bo.UserPreferenceForm) error {
	var modelID *string
	if form.DefaultModel != nil {
		if err := s.assertModelUsable(ctx, userID, *form.DefaultModel); err != nil {
			return err
		}
		modelID = form.DefaultModel
	}

	pref := &model.SysAiUserPreference{UserID: userID, DefaultModel: modelID, CreateBy: &userID, UpdateBy: &userID}
	if err := s.preferences.Upsert(ctx, pref); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "保存用户偏好失败", err)
	}
	return nil
}

// ResolveChatModel 对话默认模型解析链：请求/会话显式指定 > 用户偏好 > Agent 配置模型 > 平台默认 > 报错。
// 用户偏好行指向的模型须仍存在、启用且为 chat 类型，失效则静默跳过（不是错误，退回下一级）。
// Agent 级取会话锚定 Agent 的已发布版本快照 model_id，无已发布版本回退主表（见 resolveAgentModel）。
func (s *UserPreferenceService) ResolveChatModel(ctx context.Context, userID int64, specified *string, agentCode string) (string, error) {
	if specified != nil && *specified != "" {
		return *specified, nil
	}

	pref, err := s.preferences.FindByUserID(ctx, userID)
	if err != nil {
		return "", common.WrapBizError(common.DATABASE_ERROR, "查询用户偏好失败", err)
	}
	if pref != nil && pref.DefaultModel != nil && *pref.DefaultModel != "" {
		preferred, err := s.models.GetByModelID(ctx, *pref.DefaultModel)
		if err != nil {
			return "", common.WrapBizError(common.DATABASE_ERROR, "查询模型失败", err)
		}
		if preferred != nil && preferred.Status == 1 && preferred.ModelType == "chat" {
			return preferred.ModelID, nil
		}
	}

	agentModel, err := s.resolveAgentModel(ctx, agentCode)
	if err != nil {
		return "", err
	}
	if agentModel != nil {
		return *agentModel, nil
	}
	return s.PlatformDefaultChatModel(ctx)
}

// resolveAgentModel Agent 配置模型：优先已发布版本快照 model_id（运行面口径，版本化），
// 无已发布版本回退主表可编辑态；Agent 不存在/无模型返回 nil（跳到下一级）。
// 不做启用性判断：Agent 显式配置的模型不可用时由消息入口按统一口径报错，静默回落会掩盖配置错误。
func (s *UserPreferenceService) resolveAgentModel(ctx context.Context, agentCode string) (*string, error) {
	code := strings.TrimSpace(agentCode)
	if code == "" {
		return nil, nil
	}
	agent, err := s.agents.GetByCode(ctx, code)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询 Agent 失败", err)
	}
	if agent == nil {
		return nil, nil
	}
	version, err := s.agents.GetLatestPublished(ctx, agent.ID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询 Agent 已发布版本失败", err)
	}
	if version != nil {
		var snapshot struct {
			ModelID string `json:"model_id"`
		}
		if err := json.Unmarshal([]byte(version.Snapshot), &snapshot); err == nil {
			if modelID := strings.TrimSpace(snapshot.ModelID); modelID != "" {
				return &modelID, nil
			}
		}
	}
	if modelID := strings.TrimSpace(agent.ModelID); modelID != "" {
		return &modelID, nil
	}
	return nil, nil
}

// PlatformDefaultChatModel 平台默认对话模型：sys_dict 类型 ai_model / 键 default_model（启用项，与 python model_defaults 同口径）。
// 未配置即报 A0500，不回落代码常量：静默兜底会让"用错模型"变成难查的隐性问题。
func (s *UserPreferenceService) PlatformDefaultChatModel(ctx context.Context) (string, error) {
	options, err := s.dicts.GetByTypeCode(ctx, "ai_model")
	if err != nil {
		return "", common.WrapBizError(common.DATABASE_ERROR, "查询平台默认模型失败", err)
	}
	for _, option := range options {
		if option.Label != "default_model" {
			continue
		}
		if value := strings.TrimSpace(fmt.Sprint(option.Value)); value != "" {
			return value, nil
		}
	}
	return "", common.NewBizError(common.BUSINESS_ERROR,
		"未配置平台默认模型（sys_dict: ai_model.default_model），请先在 AI 模型管理/字典管理中配置")
}

// assertModelUsable 偏好写入前的模型校验：存在、启用、chat 类型，且不高于用户会员等级
// （等级取 HealthService.UserLevel，与 GET /ai/models/enabled 的过滤同源）。
// 四类失败共用 A0601 与同一文案，与 python ai_user_preference_service 一致（文案比错误码表更具体）。
func (s *UserPreferenceService) assertModelUsable(ctx context.Context, userID int64, modelID string) error {
	target, err := s.models.GetByModelID(ctx, modelID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询模型失败", err)
	}
	if target == nil || target.Status != 1 || target.ModelType != "chat" ||
		int(target.VipLevel) > s.health.UserLevel(ctx, userID) {
		return common.NewBizError(common.AI_MODEL_NOT_AVAILABLE, "模型不存在、未启用或当前等级不可用")
	}
	return nil
}

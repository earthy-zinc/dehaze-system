package aidomain

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/earthyzinc/dehaze-go/internal/model"
	repo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	auditlogservice "github.com/earthyzinc/dehaze-go/internal/service/audit_log"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

// reasoningDefaults 推理参数系统默认值（对齐 python agent_config_resolver.REASONING_DEFAULTS）。
var reasoningDefaults = map[string]any{
	"max_steps_react":          20,
	"max_steps_plan":           30,
	"max_steps_reflexion":      15,
	"max_iterations_reflexion": 3,
	"reflexion_threshold":      0.8,
	"max_parallel":             5,
	"tool_timeout":             60,
	"token_budget":             500000,
	"retry_max":                2,
}

// AgentConfigDefaultsVO 推理参数系统默认值对外契约（reasoningDefaults 常量键为 snake_case，
// 对外按 python AgentConfigDefaults / java AiAgentConfigDefaultsVO 的 camelCase 逐项对应）。
type AgentConfigDefaultsVO struct {
	MaxStepsReact          int     `json:"maxStepsReact"`
	MaxStepsPlan           int     `json:"maxStepsPlan"`
	MaxStepsReflexion      int     `json:"maxStepsReflexion"`
	MaxIterationsReflexion int     `json:"maxIterationsReflexion"`
	ReflexionThreshold     float64 `json:"reflexionThreshold"`
	MaxParallel            int     `json:"maxParallel"`
	ToolTimeout            int     `json:"toolTimeout"`
	TokenBudget            int     `json:"tokenBudget"`
	RetryMax               int     `json:"retryMax"`
}

// ReasoningDefaults 透出系统默认值（值只声明在 reasoningDefaults 一处，此处仅做契约映射，
// 映射错误由 TestReasoningDefaultsContractMatchesConstant 拦截，避免两处硬编码漂移）。
func ReasoningDefaults() AgentConfigDefaultsVO {
	return AgentConfigDefaultsVO{
		MaxStepsReact:          reasoningDefaults["max_steps_react"].(int),
		MaxStepsPlan:           reasoningDefaults["max_steps_plan"].(int),
		MaxStepsReflexion:      reasoningDefaults["max_steps_reflexion"].(int),
		MaxIterationsReflexion: reasoningDefaults["max_iterations_reflexion"].(int),
		ReflexionThreshold:     reasoningDefaults["reflexion_threshold"].(float64),
		MaxParallel:            reasoningDefaults["max_parallel"].(int),
		ToolTimeout:            reasoningDefaults["tool_timeout"].(int),
		TokenBudget:            reasoningDefaults["token_budget"].(int),
		RetryMax:               reasoningDefaults["retry_max"].(int),
	}
}

// guardrailDefaultsDictType 护栏系统默认字典类型。
const guardrailDefaultsDictType = "ai_guardrail_defaults"

// AgentVersionService 版本快照/回滚/版本历史业务逻辑（发布走 python 转发，门禁依赖回归评测运行时）。
type AgentVersionService struct {
	agents   *repo.AgentRepository
	auditLog *auditlogservice.AuditLogService
}

// NewAgentVersionService 构造 AgentVersionService。
func NewAgentVersionService(agents *repo.AgentRepository, auditLog *auditlogservice.AuditLogService) *AgentVersionService {
	return &AgentVersionService{agents: agents, auditLog: auditLog}
}

// buildSnapshot 序列化主表可编辑态为版本快照（含冻结的 resolved_config）。
func (s *AgentVersionService) buildSnapshot(ctx context.Context, agent *model.SysAiAgent) (map[string]any, error) {
	skills, err := s.agents.ListSkillNames(ctx, agent.ID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 Agent Skills 失败", err)
	}
	namespaces, err := s.agents.ListMcpNamespaces(ctx, agent.ID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 Agent MCP 失败", err)
	}
	links, err := s.agents.ListSubagents(ctx, agent.ID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询子 Agent 失败", err)
	}
	subagents := make([]map[string]any, 0, len(links))
	for _, link := range links {
		subagents = append(subagents, map[string]any{
			"agent_id":    link.SubagentAgentID,
			"priority":    link.Priority,
			"endpoint_id": link.EndpointID,
		})
	}
	resolvedConfig, err := s.resolveConfig(ctx, agent.Config)
	if err != nil {
		return nil, err
	}
	config := map[string]any{}
	if agent.Config != "" {
		_ = json.Unmarshal([]byte(agent.Config), &config)
	}
	var permissions any
	if agent.Permissions != "" {
		var parsed []map[string]any
		if err := json.Unmarshal([]byte(agent.Permissions), &parsed); err == nil {
			permissions = parsed
		}
	}
	return map[string]any{
		"name":            agent.Name,
		"description":     agent.Description,
		"system_prompt":   agent.SystemPrompt,
		"model_id":        agent.ModelID,
		"reasoning_mode":  agent.ReasoningMode,
		"config":          config,
		"resolved_config": resolvedConfig,
		"permissions":     permissions,
		"is_subagent":     agent.IsSubagent,
		"is_team":         agent.IsTeam,
		"is_exposed":      agent.IsExposed,
		"skills":          skills,
		"mcp_namespaces":  namespaces,
		"subagents":       subagents,
	}, nil
}

// resolveConfig 系统默认 ← Agent 配置 两级合并（护栏取 sys_dict 默认与 Agent 覆盖的逐规则合并）。
func (s *AgentVersionService) resolveConfig(ctx context.Context, agentConfigJSON string) (map[string]any, error) {
	agentConfig := map[string]any{}
	if agentConfigJSON != "" {
		_ = json.Unmarshal([]byte(agentConfigJSON), &agentConfig)
	}
	reasoning := make(map[string]any, len(reasoningDefaults))
	for key, value := range reasoningDefaults {
		reasoning[key] = value
	}
	for key, value := range agentConfig {
		if key == "guardrails" {
			continue
		}
		reasoning[key] = value
	}
	guardrails, err := s.guardrailDefaults(ctx)
	if err != nil {
		return nil, err
	}
	if overrides, ok := agentConfig["guardrails"].(map[string]any); ok {
		for rule, override := range overrides {
			overrideMap, isMap := override.(map[string]any)
			current, currentIsMap := guardrails[rule].(map[string]any)
			if isMap && currentIsMap {
				merged := make(map[string]any, len(current)+len(overrideMap))
				for key, value := range current {
					merged[key] = value
				}
				for key, value := range overrideMap {
					merged[key] = value
				}
				guardrails[rule] = merged
				continue
			}
			guardrails[rule] = override
		}
	}
	reasoning["guardrails"] = guardrails
	return reasoning, nil
}

// guardrailDefaults 读取护栏系统默认（sys_dict 点分键组装为嵌套结构）。
func (s *AgentVersionService) guardrailDefaults(ctx context.Context) (map[string]any, error) {
	values, err := s.agents.LoadDictValues(ctx, guardrailDefaultsDictType)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "读取护栏默认配置失败", err)
	}
	nested := map[string]any{}
	for key, raw := range values {
		parts := splitDotted(key)
		node := nested
		for _, part := range parts[:len(parts)-1] {
			child, ok := node[part].(map[string]any)
			if !ok {
				child = map[string]any{}
				node[part] = child
			}
			node = child
		}
		node[parts[len(parts)-1]] = coerceScalar(raw)
	}
	return nested, nil
}

// writeVersion 写入一条版本记录（草稿/已发布），版本号冲突时递增重试。
func (s *AgentVersionService) writeVersion(ctx context.Context, agent *model.SysAiAgent, operatorID *int64, changeNote string, status int) (*model.SysAiAgentVersion, error) {
	snapshot, err := s.buildSnapshot(ctx, agent)
	if err != nil {
		return nil, err
	}
	raw, err := json.Marshal(snapshot)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "序列化版本快照失败", err)
	}
	versionNo, err := s.agents.NextVersionNo(ctx, agent.ID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "计算版本号失败", err)
	}
	var note *string
	if changeNote != "" {
		note = &changeNote
	}
	for attempt := 0; attempt < 5; attempt++ {
		version := &model.SysAiAgentVersion{
			AgentID:    agent.ID,
			VersionNo:  versionNo,
			Snapshot:   string(raw),
			Status:     status,
			ChangeNote: note,
			OperatorID: operatorID,
		}
		if err := s.agents.CreateVersion(ctx, version); err != nil {
			versionNo++
			continue
		}
		return version, nil
	}
	return nil, common.NewBizError(common.DATA_EXISTS, "版本号并发冲突，请重试发布")
}

// Rollback 回滚到历史已发布版本（不覆盖历史，生成新已发布版本）。
func (s *AgentVersionService) Rollback(ctx context.Context, agentID int64, versionNo int, operatorID int64) (int, error) {
	agent, err := s.agents.GetByID(ctx, agentID)
	if err != nil {
		return 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 Agent 失败", err)
	}
	if agent == nil {
		return 0, common.NewBizError(common.RESOURCE_NOT_FOUND, "Agent 不存在")
	}
	target, err := s.agents.GetVersion(ctx, agentID, versionNo)
	if err != nil {
		return 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询版本失败", err)
	}
	if target == nil {
		return 0, common.NewBizError(common.RESOURCE_NOT_FOUND, "回滚目标版本不存在")
	}
	if target.Status != 2 {
		return 0, common.NewBizError(common.DATA_STATE_NOT_ALLOW, "仅可回滚到已发布版本")
	}
	var snapshot map[string]any
	if err := json.Unmarshal([]byte(target.Snapshot), &snapshot); err != nil {
		return 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "解析版本快照失败", err)
	}

	fields := map[string]any{"update_by": operatorID}
	applySnapshotString(fields, snapshot, "name", "name")
	applySnapshotString(fields, snapshot, "description", "description")
	applySnapshotString(fields, snapshot, "system_prompt", "system_prompt")
	applySnapshotString(fields, snapshot, "model_id", "model_id")
	applySnapshotString(fields, snapshot, "reasoning_mode", "reasoning_mode")
	// 快照里 config/permissions 为空（python 侧写 None→NULL）时必须落 NULL：
	// 这里走 map 型更新，写空串会触发 MySQL 3140
	if value, ok := snapshot["config"]; ok {
		fields["config"] = jsonColumnValue(value)
	}
	if value, ok := snapshot["permissions"]; ok {
		fields["permissions"] = jsonColumnValue(value)
	}
	for _, key := range []string{"is_subagent", "is_team", "is_exposed"} {
		if value, ok := snapshot[key]; ok {
			fields[key] = value
		}
	}
	if err := s.agents.UpdateFields(ctx, agentID, fields); err != nil {
		return 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "回滚 Agent 失败", err)
	}
	if err := s.agents.ReplaceSkills(ctx, agentID, snapshotStrings(snapshot, "skills")); err != nil {
		return 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "恢复 Skills 失败", err)
	}
	if err := s.agents.ReplaceMcpNamespaces(ctx, agentID, snapshotStrings(snapshot, "mcp_namespaces")); err != nil {
		return 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "恢复 MCP 失败", err)
	}
	if err := s.agents.ReplaceSubagents(ctx, agentID, snapshotSubagents(snapshot)); err != nil {
		return 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "恢复子 Agent 失败", err)
	}

	reloaded, err := s.agents.GetByID(ctx, agentID)
	if err != nil || reloaded == nil {
		return 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 Agent 失败", err)
	}
	if err := s.agents.DemotePublished(ctx, agentID); err != nil {
		return 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "更新版本状态失败", err)
	}
	version, err := s.writeVersion(ctx, reloaded, &operatorID, fmt.Sprintf("回滚自 v%d", versionNo), 2)
	if err != nil {
		return 0, err
	}
	invalidateCacheKeys(ctx,
		fmt.Sprintf(agentPublishedKeyFmt, agentID),
		fmt.Sprintf(agentDetailKeyFmt, reloaded.AgentCode),
		fmt.Sprintf(agentSkillKeyFmt, agentID),
		fmt.Sprintf(agentMcpKeyFmt, agentID),
		fmt.Sprintf(agentSubagentKeyFmt, agentID),
		agentEnabledListKey,
	)
	if s.auditLog != nil {
		s.auditLog.RecordAuditAsync(ctx, operatorID, "ai_agent", agentID, "rollback", "ai_agent", nil,
			map[string]any{"from_version_no": versionNo, "to_version_no": version.VersionNo}, "", "")
	}
	return version.VersionNo, nil
}

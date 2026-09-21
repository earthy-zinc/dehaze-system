package aidomain

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	repo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	auditlogservice "github.com/earthyzinc/dehaze-go/internal/service/audit_log"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

// defaultAgentCode 默认 Agent 编码（系统预置且不可删除）。
const defaultAgentCode = "default"

var validReasoningModes = map[string]struct{}{
	"auto": {}, "direct": {}, "react": {}, "plan_execute": {}, "reflexion": {},
}

// AgentListItemVO Agent 列表项。
type AgentListItemVO struct {
	ID            int64    `json:"id"`
	AgentCode     string   `json:"agentCode"`
	Name          string   `json:"name"`
	Description   string   `json:"description"`
	ModelID       string   `json:"modelId"`
	ReasoningMode string   `json:"reasoningMode"`
	IsSubagent    int      `json:"isSubagent"`
	IsTeam        int      `json:"isTeam"`
	IsExposed     int      `json:"isExposed"`
	Tags          []string `json:"tags"`
	Status        int      `json:"status"`
	SortOrder     int      `json:"sortOrder"`
	CreateTime    string   `json:"createTime,omitempty"`
	SkillCount    int64    `json:"skillCount"`
	McpCount      int64    `json:"mcpCount"`
	SubAgentCount int64    `json:"subAgentCount"`
}

// AgentDetailVO Agent 详情。
type AgentDetailVO struct {
	AgentListItemVO
	SystemPrompt  string           `json:"systemPrompt,omitempty"`
	Config        json.RawMessage  `json:"config,omitempty"`
	Permissions   json.RawMessage  `json:"permissions,omitempty"`
	Skills        []string         `json:"skills"`
	McpNamespaces []string         `json:"mcpNamespaces"`
	Subagents     []SubAgentItemVO `json:"subagents"`
}

// SubAgentItemVO 子 Agent 关联项。
type SubAgentItemVO struct {
	AgentID     int64  `json:"agentId"`
	AgentName   string `json:"agentName"`
	AgentCode   string `json:"agentCode"`
	Description string `json:"description"`
	EndpointID  *int64 `json:"endpointId,omitempty"`
	Priority    int    `json:"priority"`
}

// AgentCreateForm 创建 Agent 表单（约束逐项对齐 python AgentCreate）。
type AgentCreateForm struct {
	AgentCode     string          `json:"agentCode" binding:"required,min=1,max=64"`
	Name          string          `json:"name" binding:"required,min=1,max=128"`
	Description   string          `json:"description" binding:"max=512"`
	SystemPrompt  *string         `json:"systemPrompt"`
	ModelID       string          `json:"modelId" binding:"required,min=1,max=64"`
	ReasoningMode string          `json:"reasoningMode" binding:"omitempty,oneof=auto direct react plan_execute reflexion"`
	Config        json.RawMessage `json:"config"`
	IsSubagent    bool            `json:"isSubagent"`
	IsTeam        bool            `json:"isTeam"`
	IsExposed     bool            `json:"isExposed"`
	Permissions   json.RawMessage `json:"permissions"`
	Tags          []string        `json:"tags"`
	SortOrder     int             `json:"sortOrder" binding:"gte=0"`
	Status        *int            `json:"status" binding:"omitempty,oneof=0 1"`
}

// AgentUpdateForm 更新 Agent 表单（约束逐项对齐 python AgentUpdate；指针区分未传字段）。
type AgentUpdateForm struct {
	Name          *string         `json:"name" binding:"omitempty,min=1,max=128"`
	Description   *string         `json:"description" binding:"omitempty,max=512"`
	SystemPrompt  *string         `json:"systemPrompt"`
	ModelID       *string         `json:"modelId" binding:"omitempty,min=1,max=64"`
	ReasoningMode *string         `json:"reasoningMode" binding:"omitempty,oneof=auto direct react plan_execute reflexion"`
	Config        json.RawMessage `json:"config"`
	IsSubagent    *bool           `json:"isSubagent"`
	IsTeam        *bool           `json:"isTeam"`
	IsExposed     *bool           `json:"isExposed"`
	Permissions   json.RawMessage `json:"permissions"`
	Tags          []string        `json:"tags"`
	SortOrder     *int            `json:"sortOrder" binding:"omitempty,gte=0"`
}

// AgentSkillsForm 覆盖式设置 Skills。
type AgentSkillsForm struct {
	Skills []string `json:"skills"`
}

// AgentMcpForm 覆盖式设置 MCP 命名空间（python `AgentMcpForm` 为纯 BaseModel，wire 字段 `mcp_namespaces`）。
type AgentMcpForm struct {
	McpNamespaces []string `json:"mcp_namespaces"`
}

// AgentSubAgentItemForm 子 Agent 关联项（python `AgentSubAgentItem` 为纯 BaseModel，wire 字段 snake_case）。
type AgentSubAgentItemForm struct {
	AgentID    int64  `json:"agent_id"`
	EndpointID *int64 `json:"endpoint_id"`
	Priority   int    `json:"priority"`
}

// AgentSubAgentsForm 覆盖式设置子 Agent。
type AgentSubAgentsForm struct {
	Subagents []AgentSubAgentItemForm `json:"subagents"`
}

// AgentService Agent 管理业务逻辑。
type AgentService struct {
	agents   *repo.AgentRepository
	eval     *repo.EvalRepository
	auditLog *auditlogservice.AuditLogService
}

// NewAgentService 构造 AgentService。
func NewAgentService(agents *repo.AgentRepository, eval *repo.EvalRepository, auditLog *auditlogservice.AuditLogService) *AgentService {
	return &AgentService{agents: agents, eval: eval, auditLog: auditLog}
}

// List Agent 列表（管理端分页全量；非管理者的可选列表走 ListEnabled）。
func (s *AgentService) List(ctx context.Context, page, size int, keyword string, status *int, agentType string) (*vo.PageResult[AgentListItemVO], error) {
	items, total, err := s.agents.Paginate(ctx, page, size, keyword, status, agentType)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 Agent 列表失败", err)
	}
	list := toAgentListVOs(items)
	if err := s.attachLinkCounts(ctx, items, list); err != nil {
		return nil, err
	}
	return &vo.PageResult[AgentListItemVO]{List: list, Total: total}, nil
}

// ListEnabled 可选 Agent 列表。
func (s *AgentService) ListEnabled(ctx context.Context) ([]AgentListItemVO, error) {
	items, err := s.agents.ListEnabled(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 Agent 列表失败", err)
	}
	list := toAgentListVOs(items)
	if err := s.attachLinkCounts(ctx, items, list); err != nil {
		return nil, err
	}
	return list, nil
}

// GetDetail Agent 详情。
func (s *AgentService) GetDetail(ctx context.Context, id int64) (*AgentDetailVO, error) {
	agent, err := s.requireAgent(ctx, id)
	if err != nil {
		return nil, err
	}
	return s.buildDetail(ctx, agent)
}

// Create 创建 Agent。
func (s *AgentService) Create(ctx context.Context, form *AgentCreateForm) (*AgentDetailVO, error) {
	if strings.TrimSpace(form.AgentCode) == "" || strings.TrimSpace(form.Name) == "" || strings.TrimSpace(form.ModelID) == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "Agent 编码/名称/模型标识不能为空")
	}
	mode := form.ReasoningMode
	if mode == "" {
		mode = "auto"
	}
	if _, ok := validReasoningModes[mode]; !ok {
		return nil, common.NewBizError(common.PARAM_ERROR, "推理范式取值非法")
	}
	existing, err := s.agents.GetByCode(ctx, form.AgentCode)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 Agent 失败", err)
	}
	if existing != nil {
		return nil, common.NewBizError(common.DATA_EXISTS, "Agent 编码已存在")
	}
	status := 1
	if form.Status != nil {
		status = *form.Status
	}
	agent := &model.SysAiAgent{
		AgentCode:     form.AgentCode,
		Name:          form.Name,
		Description:   form.Description,
		SystemPrompt:  form.SystemPrompt,
		ModelID:       form.ModelID,
		ReasoningMode: mode,
		Config:        normalizeConfigJSON(form.Config),
		IsSubagent:    boolToInt(form.IsSubagent),
		IsTeam:        boolToInt(form.IsTeam),
		IsExposed:     boolToInt(form.IsExposed),
		Permissions:   normalizeNullableJSON(form.Permissions),
		Tags:          marshalJSON(form.Tags),
		SortOrder:     form.SortOrder,
		Status:        status,
	}
	if err := s.agents.Create(ctx, agent); err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "创建 Agent 失败", err)
	}
	invalidateCacheKeys(ctx, agentEnabledListKey)
	return s.buildDetail(ctx, agent)
}

// Update 更新 Agent 可编辑态。
func (s *AgentService) Update(ctx context.Context, id int64, form *AgentUpdateForm) (*AgentDetailVO, error) {
	agent, err := s.requireAgent(ctx, id)
	if err != nil {
		return nil, err
	}
	fields := map[string]any{}
	if form.Name != nil {
		fields["name"] = *form.Name
		agent.Name = *form.Name
	}
	if form.Description != nil {
		fields["description"] = *form.Description
		agent.Description = *form.Description
	}
	if form.SystemPrompt != nil {
		fields["system_prompt"] = *form.SystemPrompt
		agent.SystemPrompt = form.SystemPrompt
	}
	if form.ModelID != nil {
		fields["model_id"] = *form.ModelID
		agent.ModelID = *form.ModelID
	}
	if form.ReasoningMode != nil {
		if _, ok := validReasoningModes[*form.ReasoningMode]; !ok {
			return nil, common.NewBizError(common.PARAM_ERROR, "推理范式取值非法")
		}
		fields["reasoning_mode"] = *form.ReasoningMode
		agent.ReasoningMode = *form.ReasoningMode
	}
	if form.Config != nil {
		config := normalizeConfigJSON(form.Config)
		fields["config"] = config
		agent.Config = config
	}
	if form.IsSubagent != nil {
		fields["is_subagent"] = boolToInt(*form.IsSubagent)
		agent.IsSubagent = boolToInt(*form.IsSubagent)
	}
	if form.IsTeam != nil {
		fields["is_team"] = boolToInt(*form.IsTeam)
		agent.IsTeam = boolToInt(*form.IsTeam)
	}
	if form.IsExposed != nil {
		fields["is_exposed"] = boolToInt(*form.IsExposed)
		agent.IsExposed = boolToInt(*form.IsExposed)
	}
	if form.Permissions != nil {
		permissions := normalizeNullableJSON(form.Permissions)
		fields["permissions"] = permissions
		agent.Permissions = permissions
	}
	if form.Tags != nil {
		tags := marshalJSON(form.Tags)
		fields["tags"] = tags
		agent.Tags = tags
	}
	if form.SortOrder != nil {
		fields["sort_order"] = *form.SortOrder
		agent.SortOrder = *form.SortOrder
	}
	if len(fields) > 0 {
		if err := s.agents.UpdateFields(ctx, id, fields); err != nil {
			return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "更新 Agent 失败", err)
		}
	}
	s.invalidateAgentCaches(ctx, agent)
	return s.buildDetail(ctx, agent)
}

// SetStatus 启停 Agent。
func (s *AgentService) SetStatus(ctx context.Context, id int64, status int) error {
	agent, err := s.requireAgent(ctx, id)
	if err != nil {
		return err
	}
	if status != 0 && status != 1 {
		return common.NewBizError(common.PARAM_ERROR, "状态取值非法")
	}
	if err := s.agents.UpdateFields(ctx, id, map[string]any{"status": status}); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "启停 Agent 失败", err)
	}
	agent.Status = status
	s.invalidateAgentCaches(ctx, agent)
	return nil
}

// Delete 删除 Agent（级联清理评测资产）。
func (s *AgentService) Delete(ctx context.Context, id, operatorID int64) error {
	agent, err := s.requireAgent(ctx, id)
	if err != nil {
		return err
	}
	if agent.AgentCode == defaultAgentCode {
		return common.NewBizError(common.OPERATION_NOT_ALLOW, "默认 Agent 不可删除")
	}
	conversationRefs, err := s.agents.CountConversationReferences(ctx, agent.AgentCode)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询会话引用失败", err)
	}
	if conversationRefs > 0 {
		return common.NewBizError(common.DATA_BIND_EXISTS,
			fmt.Sprintf("存在 %d 个会话正在使用该 Agent，请先解绑", conversationRefs))
	}
	subagentRefs, err := s.agents.CountSubagentReferences(ctx, id)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询子 Agent 引用失败", err)
	}
	if subagentRefs > 0 {
		return common.NewBizError(common.DATA_BIND_EXISTS,
			fmt.Sprintf("该 Agent 被 %d 个 Agent 作为子 Agent 引用，请先解绑", subagentRefs))
	}

	// 评测资产按 agent_id 挂载，Agent 软删后失去清理入口，故一并清理：
	// 样本随评测集物理删除，评测集软删，执行记录为只追加轨迹随 Agent 物理清理。
	datasets, err := s.eval.ListDatasetsByAgent(ctx, id)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询评测集失败", err)
	}
	datasetIDs := make([]int64, 0, len(datasets))
	for _, d := range datasets {
		datasetIDs = append(datasetIDs, d.ID)
	}
	sampleCount := int64(0)
	if len(datasetIDs) > 0 {
		if sampleCount, err = s.eval.DeleteSamplesByDatasets(ctx, datasetIDs); err != nil {
			return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "清理评测样本失败", err)
		}
		if err = s.eval.SoftDeleteDatasets(ctx, datasetIDs, operatorID); err != nil {
			return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "清理评测集失败", err)
		}
	}
	runCount, err := s.eval.DeleteRunsByAgent(ctx, id)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "清理评测记录失败", err)
	}
	if err := s.agents.SoftDeleteByIDs(ctx, []int64{id}, operatorID); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "删除 Agent 失败", err)
	}
	s.invalidateAgentCaches(ctx, agent)
	if s.auditLog != nil {
		s.auditLog.RecordAuditAsync(ctx, operatorID, "ai_agent", id, "delete", "ai_agent",
			map[string]any{"agent_code": agent.AgentCode, "name": agent.Name},
			map[string]any{
				"eval_datasets_soft_deleted": len(datasetIDs),
				"eval_samples_deleted":       sampleCount,
				"eval_runs_deleted":          runCount,
			}, "", "")
	}
	return nil
}

// Copy 复制 Agent（基本信息与配置，不复制关联关系）。
func (s *AgentService) Copy(ctx context.Context, id int64, newCode string) (*AgentDetailVO, error) {
	source, err := s.requireAgent(ctx, id)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(newCode) == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "新 Agent 编码不能为空")
	}
	existing, err := s.agents.GetByCode(ctx, newCode)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 Agent 失败", err)
	}
	if existing != nil {
		return nil, common.NewBizError(common.DATA_EXISTS, "Agent 编码已存在")
	}
	copied := &model.SysAiAgent{
		AgentCode:     newCode,
		Name:          source.Name,
		Description:   source.Description,
		SystemPrompt:  source.SystemPrompt,
		ModelID:       source.ModelID,
		ReasoningMode: source.ReasoningMode,
		Config:        source.Config,
		IsSubagent:    source.IsSubagent,
		IsTeam:        source.IsTeam,
		IsExposed:     source.IsExposed,
		Permissions:   source.Permissions,
		Tags:          source.Tags,
		SortOrder:     source.SortOrder,
		Status:        1,
	}
	if err := s.agents.Create(ctx, copied); err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "复制 Agent 失败", err)
	}
	invalidateCacheKeys(ctx, agentEnabledListKey)
	return s.buildDetail(ctx, copied)
}

// SetSkills 覆盖式设置 Skills（引用完整性校验）。
func (s *AgentService) SetSkills(ctx context.Context, id int64, skillNames []string) error {
	agent, err := s.requireAgent(ctx, id)
	if err != nil {
		return err
	}
	if len(skillNames) > 0 {
		existing, err := s.agents.ListExistingSkillNames(ctx, skillNames)
		if err != nil {
			return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 Skill 失败", err)
		}
		if missing := diffSet(skillNames, existing); len(missing) > 0 {
			return common.NewBizError(common.RESOURCE_NOT_FOUND,
				fmt.Sprintf("以下 Skill 不存在: %s", strings.Join(head(missing, 5), ", ")))
		}
	}
	if err := s.agents.ReplaceSkills(ctx, id, skillNames); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "设置 Skills 失败", err)
	}
	s.invalidateAgentCaches(ctx, agent)
	return nil
}

// SetMcp 覆盖式设置 MCP 命名空间（引用完整性校验）。
func (s *AgentService) SetMcp(ctx context.Context, id int64, namespaces []string) error {
	agent, err := s.requireAgent(ctx, id)
	if err != nil {
		return err
	}
	if len(namespaces) > 0 {
		registered, err := s.agents.ListRegisteredNamespaces(ctx, namespaces)
		if err != nil {
			return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 MCP 命名空间失败", err)
		}
		if missing := diffSet(namespaces, registered); len(missing) > 0 {
			return common.NewBizError(common.RESOURCE_NOT_FOUND,
				fmt.Sprintf("以下 MCP 命名空间未注册: %s", strings.Join(head(missing, 5), ", ")))
		}
	}
	if err := s.agents.ReplaceMcpNamespaces(ctx, id, namespaces); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "设置 MCP 命名空间失败", err)
	}
	s.invalidateAgentCaches(ctx, agent)
	return nil
}

// SetSubagents 覆盖式设置子 Agent（自引用/环校验）。
func (s *AgentService) SetSubagents(ctx context.Context, id int64, form *AgentSubAgentsForm) error {
	agent, err := s.requireAgent(ctx, id)
	if err != nil {
		return err
	}
	items := make([]model.SysAiAgentSubagent, 0, len(form.Subagents))
	childIDs := make([]int64, 0, len(form.Subagents))
	for _, item := range form.Subagents {
		if item.AgentID == id {
			return common.NewBizError(common.PARAM_ERROR, "子 Agent 不能是自身")
		}
		childIDs = append(childIDs, item.AgentID)
		items = append(items, model.SysAiAgentSubagent{
			SubagentAgentID: item.AgentID,
			EndpointID:      item.EndpointID,
			Priority:        item.Priority,
		})
	}
	if len(form.Subagents) != len(uniqueIDs(childIDs)) {
		return common.NewBizError(common.PARAM_ERROR, "子 Agent 列表存在重复项")
	}
	if len(childIDs) > 0 {
		found, err := s.agents.GetByIDs(ctx, childIDs)
		if err != nil {
			return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询子 Agent 失败", err)
		}
		existing := make([]int64, 0, len(found))
		for _, item := range found {
			existing = append(existing, item.ID)
		}
		if missing := diffInt64(childIDs, existing); len(missing) > 0 {
			return common.NewBizError(common.RESOURCE_NOT_FOUND,
				fmt.Sprintf("以下子 Agent 不存在: %s", joinInt64(headInt64(missing, 5))))
		}
	}
	if err := s.ensureAcyclic(ctx, id, childIDs); err != nil {
		return err
	}
	if err := s.agents.ReplaceSubagents(ctx, id, items); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "设置子 Agent 失败", err)
	}
	s.invalidateAgentCaches(ctx, agent)
	return nil
}

// ensureAcyclic 校验绑定后不形成子 Agent 环（环会让推理期子 Agent 展开无限递归）。
func (s *AgentService) ensureAcyclic(ctx context.Context, agentID int64, childIDs []int64) error {
	path := []int64{}
	settled := map[int64]struct{}{}

	var walk func(node int64) error
	walk = func(node int64) error {
		if _, ok := settled[node]; ok {
			return nil
		}
		for idx, item := range path {
			if item == node {
				cycle := append(append([]int64{}, path[idx:]...), node)
				return common.NewBizError(common.PARAM_ERROR, "子 Agent 绑定存在环: "+joinInt64(cycle))
			}
		}
		path = append(path, node)
		children := childIDs
		if node != agentID {
			links, err := s.agents.ListSubagents(ctx, node)
			if err != nil {
				return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询子 Agent 失败", err)
			}
			children = make([]int64, 0, len(links))
			for _, link := range links {
				children = append(children, link.SubagentAgentID)
			}
		}
		for _, child := range children {
			if err := walk(child); err != nil {
				return err
			}
		}
		path = path[:len(path)-1]
		settled[node] = struct{}{}
		return nil
	}
	return walk(agentID)
}

func (s *AgentService) requireAgent(ctx context.Context, id int64) (*model.SysAiAgent, error) {
	agent, err := s.agents.GetByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 Agent 失败", err)
	}
	if agent == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "Agent 不存在")
	}
	return agent, nil
}

func (s *AgentService) buildDetail(ctx context.Context, agent *model.SysAiAgent) (*AgentDetailVO, error) {
	skills, err := s.agents.ListSkillNames(ctx, agent.ID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 Agent Skills 失败", err)
	}
	namespaces, err := s.agents.ListMcpNamespaces(ctx, agent.ID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 Agent MCP 失败", err)
	}
	subagents, err := s.agents.ListSubagentItems(ctx, agent.ID)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询子 Agent 失败", err)
	}
	detail := &AgentDetailVO{
		AgentListItemVO: toAgentListVO(agent),
		SystemPrompt:    derefString(agent.SystemPrompt),
		Config:          rawJSON(agent.Config),
		Permissions:     rawJSON(agent.Permissions),
		Skills:          skills,
		McpNamespaces:   namespaces,
		Subagents:       make([]SubAgentItemVO, 0, len(subagents)),
	}
	list := []AgentListItemVO{detail.AgentListItemVO}
	if err := s.attachLinkCounts(ctx, []model.SysAiAgent{*agent}, list); err != nil {
		return nil, err
	}
	detail.AgentListItemVO = list[0]
	for _, item := range subagents {
		detail.Subagents = append(detail.Subagents, SubAgentItemVO{
			AgentID:     item.AgentID,
			AgentName:   item.AgentName,
			AgentCode:   item.AgentCode,
			Description: item.Description,
			EndpointID:  item.EndpointID,
			Priority:    item.Priority,
		})
	}
	return detail, nil
}

func (s *AgentService) attachLinkCounts(ctx context.Context, agents []model.SysAiAgent, list []AgentListItemVO) error {
	if len(agents) == 0 {
		return nil
	}
	ids := make([]int64, 0, len(agents))
	for _, agent := range agents {
		ids = append(ids, agent.ID)
	}
	skillCounts, err := s.agents.CountLinksByAgentIDs(ctx, "sys_ai_agent_skill", "skill_name", ids)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "统计 Skills 失败", err)
	}
	mcpCounts, err := s.agents.CountLinksByAgentIDs(ctx, "sys_ai_agent_mcp", "mcp_namespace", ids)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "统计 MCP 失败", err)
	}
	subCounts, err := s.agents.CountSubagentsByAgentIDs(ctx, ids)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "统计子 Agent 失败", err)
	}
	for i := range list {
		list[i].SkillCount = skillCounts[list[i].ID]
		list[i].McpCount = mcpCounts[list[i].ID]
		list[i].SubAgentCount = subCounts[list[i].ID]
	}
	return nil
}

// invalidateAgentCaches 失效 Agent 相关缓存（提交后调用）。
func (s *AgentService) invalidateAgentCaches(ctx context.Context, agent *model.SysAiAgent) {
	invalidateCacheKeys(ctx,
		fmt.Sprintf(agentDetailKeyFmt, agent.AgentCode),
		fmt.Sprintf(agentSkillKeyFmt, agent.ID),
		fmt.Sprintf(agentMcpKeyFmt, agent.ID),
		fmt.Sprintf(agentSubagentKeyFmt, agent.ID),
		fmt.Sprintf(agentPublishedKeyFmt, agent.ID),
		agentEnabledListKey,
	)
}

func toAgentListVO(agent *model.SysAiAgent) AgentListItemVO {
	tags := []string{}
	if agent.Tags != "" {
		_ = json.Unmarshal([]byte(agent.Tags), &tags)
	}
	return AgentListItemVO{
		ID:            agent.ID,
		AgentCode:     agent.AgentCode,
		Name:          agent.Name,
		Description:   agent.Description,
		ModelID:       agent.ModelID,
		ReasoningMode: agent.ReasoningMode,
		IsSubagent:    agent.IsSubagent,
		IsTeam:        agent.IsTeam,
		IsExposed:     agent.IsExposed,
		Tags:          tags,
		Status:        agent.Status,
		SortOrder:     agent.SortOrder,
		CreateTime:    formatTime(agent.CreateTime),
	}
}

func toAgentListVOs(agents []model.SysAiAgent) []AgentListItemVO {
	result := make([]AgentListItemVO, 0, len(agents))
	for i := range agents {
		result = append(result, toAgentListVO(&agents[i]))
	}
	return result
}

// normalizeConfigJSON 去掉 null 值（对齐 python model_dump(exclude_none=True)）。
func normalizeConfigJSON(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var parsed map[string]any
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return string(raw)
	}
	return marshalJSON(dropNilValues(parsed))
}

func normalizeNullableJSON(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	return string(raw)
}

func dropNilValues(input map[string]any) map[string]any {
	result := make(map[string]any, len(input))
	for key, value := range input {
		if value == nil {
			continue
		}
		if nested, ok := value.(map[string]any); ok {
			result[key] = dropNilValues(nested)
			continue
		}
		result[key] = value
	}
	return result
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

func diffSet(input, existing []string) []string {
	set := make(map[string]struct{}, len(existing))
	for _, item := range existing {
		set[item] = struct{}{}
	}
	missing := []string{}
	for _, item := range input {
		if _, ok := set[item]; !ok {
			missing = append(missing, item)
		}
	}
	return missing
}

func diffInt64(input, existing []int64) []int64 {
	set := make(map[int64]struct{}, len(existing))
	for _, item := range existing {
		set[item] = struct{}{}
	}
	missing := []int64{}
	for _, item := range input {
		if _, ok := set[item]; !ok {
			missing = append(missing, item)
		}
	}
	return missing
}

func head(input []string, n int) []string {
	if len(input) > n {
		return input[:n]
	}
	return input
}

func headInt64(input []int64, n int) []int64 {
	if len(input) > n {
		return input[:n]
	}
	return input
}

func uniqueIDs(ids []int64) []int64 {
	seen := make(map[int64]struct{}, len(ids))
	result := make([]int64, 0, len(ids))
	for _, id := range ids {
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		result = append(result, id)
	}
	return result
}

func joinInt64(ids []int64) string {
	parts := make([]string, 0, len(ids))
	for _, id := range ids {
		parts = append(parts, strconv.FormatInt(id, 10))
	}
	return strings.Join(parts, "→")
}

package api

import (
	"strconv"

	aidomain "github.com/earthyzinc/dehaze-go/internal/service/aidomain"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

// agentManagePermission Agent 管理权限标识。
const agentManagePermission = "ai:agent:manage"

// AiAgentApi 智能体管理/A2A 端点（A 类，版本快照与发布门禁）。
type AiAgentApi struct {
	agents    *aidomain.AgentService
	versions  *aidomain.AgentVersionService
	endpoints *aidomain.EndpointService
}

// NewAiAgentApi 构造 AiAgentApi。
func NewAiAgentApi(
	agents *aidomain.AgentService,
	versions *aidomain.AgentVersionService,
	endpoints *aidomain.EndpointService,
) *AiAgentApi {
	return &AiAgentApi{agents: agents, versions: versions, endpoints: endpoints}
}

// isAgentManager 是否为 Agent 管理者（ROOT 或持有 ai:agent:manage）。
func isAgentManager(c *gin.Context) bool {
	if security.IsRoot(c) {
		return true
	}
	has, err := security.HasAnyPermission(c, agentManagePermission)
	return err == nil && has
}

// ensureAgentManager 版本管理读接口权限校验（快照含完整提示词/权限/配置，仅管理端可见）。
func ensureAgentManager(c *gin.Context) bool {
	if isAgentManager(c) {
		return true
	}
	_ = c.Error(common.NewBizError(common.ACCESS_UNAUTHORIZED, "无权访问版本管理信息"))
	return false
}

// ListAgents Agent 列表（管理者分页全量，普通用户返回启用列表）。
func (a *AiAgentApi) ListAgents(c *gin.Context) {
	if !isAgentManager(c) {
		result, err := a.agents.ListEnabled(c.Request.Context())
		if err != nil {
			_ = c.Error(err)
			return
		}
		common.OkWithDetailed(gin.H{"list": result, "total": len(result)}, "一切ok", c)
		return
	}
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	status, ok := parseRangedOptionalInt(c.Query("status"), 0, 1)
	if !ok {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "status 仅支持 0/1"))
		return
	}
	result, err := a.agents.List(c.Request.Context(), pageNum, pageSize, c.Query("keyword"), status, c.Query("type"))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListEnabledAgents 可选 Agent 列表。
func (a *AiAgentApi) ListEnabledAgents(c *gin.Context) {
	result, err := a.agents.ListEnabled(c.Request.Context())
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetAgentConfigDefaults 推理参数系统默认值（登录即可读；Agent 配置表单"空值继承系统默认"提示用）。
func (a *AiAgentApi) GetAgentConfigDefaults(c *gin.Context) {
	common.OkWithData(aidomain.ReasoningDefaults(), c)
}

// CreateAgent 创建 Agent。
func (a *AiAgentApi) CreateAgent(c *gin.Context) {
	var form aidomain.AgentCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.agents.Create(c.Request.Context(), &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetAgent Agent 详情。
func (a *AiAgentApi) GetAgent(c *gin.Context) {
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	result, err := a.agents.GetDetail(c.Request.Context(), id)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdateAgent 更新 Agent。
func (a *AiAgentApi) UpdateAgent(c *gin.Context) {
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	var form aidomain.AgentUpdateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.agents.Update(c.Request.Context(), id, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// DeleteAgent 删除 Agent。
func (a *AiAgentApi) DeleteAgent(c *gin.Context) {
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	operatorID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	if err := a.agents.Delete(c.Request.Context(), id, operatorID); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("一切ok", c)
}

// SetAgentStatus 启停 Agent。
func (a *AiAgentApi) SetAgentStatus(c *gin.Context) {
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	var form struct {
		Status *int `json:"status"`
	}
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	if form.Status == nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "status 不能为空"))
		return
	}
	if err := a.agents.SetStatus(c.Request.Context(), id, *form.Status); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("一切ok", c)
}

// CopyAgent 复制 Agent。
func (a *AiAgentApi) CopyAgent(c *gin.Context) {
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	var form struct {
		// python `AgentCopyForm` 为纯 BaseModel，wire 字段为 `agent_code`
		AgentCode string `json:"agent_code" binding:"required,min=1,max=64"`
	}
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.agents.Copy(c.Request.Context(), id, form.AgentCode)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// SetAgentSkills 设置 Skills（覆盖式）。
func (a *AiAgentApi) SetAgentSkills(c *gin.Context) {
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	var form aidomain.AgentSkillsForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	if err := a.agents.SetSkills(c.Request.Context(), id, form.Skills); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("一切ok", c)
}

// SetAgentMcps 设置 MCP 命名空间（覆盖式）。
func (a *AiAgentApi) SetAgentMcps(c *gin.Context) {
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	var form aidomain.AgentMcpForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	if err := a.agents.SetMcp(c.Request.Context(), id, form.McpNamespaces); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("一切ok", c)
}

// SetAgentSubagents 设置子 Agent（覆盖式）。
func (a *AiAgentApi) SetAgentSubagents(c *gin.Context) {
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	var form aidomain.AgentSubAgentsForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	if err := a.agents.SetSubagents(c.Request.Context(), id, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("一切ok", c)
}

// ListAgentVersions 版本历史。
func (a *AiAgentApi) ListAgentVersions(c *gin.Context) {
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	if !ensureAgentManager(c) {
		return
	}
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	result, err := a.versions.ListVersions(c.Request.Context(), id, pageNum, pageSize)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// DiffAgentVersions 版本差异对比。
func (a *AiAgentApi) DiffAgentVersions(c *gin.Context) {
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	base, errBase := strconv.Atoi(c.Query("base"))
	target, errTarget := strconv.Atoi(c.Query("target"))
	if errBase != nil || errTarget != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "base/target 参数非法"))
		return
	}
	if !ensureAgentManager(c) {
		return
	}
	result, err := a.versions.DiffVersions(c.Request.Context(), id, base, target)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetAgentVersionDetail 版本快照详情。
func (a *AiAgentApi) GetAgentVersionDetail(c *gin.Context) {
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	versionNo, err := strconv.Atoi(c.Param("versionNo"))
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "版本号格式不正确"))
		return
	}
	if !ensureAgentManager(c) {
		return
	}
	result, err := a.versions.GetVersionDetail(c.Request.Context(), id, versionNo)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// RollbackAgent 回滚到历史版本。
func (a *AiAgentApi) RollbackAgent(c *gin.Context) {
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	versionNo, err := strconv.Atoi(c.Param("versionNo"))
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "版本号格式不正确"))
		return
	}
	operatorID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	newVersionNo, err := a.versions.Rollback(c.Request.Context(), id, versionNo, operatorID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	// 字段名与 python rollback 的 dict 返回一致（SDK VersionResult.version_no），勿改 camelCase
	common.OkWithData(gin.H{"version_no": newVersionNo}, c)
}

// ── 外部 A2A 端点 ─────────────────────────────────────────────

// CreateEndpoint 注册外部 A2A 端点。
func (a *AiAgentApi) CreateEndpoint(c *gin.Context) {
	var form aidomain.EndpointCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.endpoints.Create(c.Request.Context(), &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdateEndpoint 更新端点。
func (a *AiAgentApi) UpdateEndpoint(c *gin.Context) {
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	var form aidomain.EndpointUpdateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.endpoints.Update(c.Request.Context(), id, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// DeleteEndpoint 删除端点。
func (a *AiAgentApi) DeleteEndpoint(c *gin.Context) {
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	operatorID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	if err := a.endpoints.Delete(c.Request.Context(), id, operatorID); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("一切ok", c)
}

// ListEndpoints 端点分页列表。
func (a *AiAgentApi) ListEndpoints(c *gin.Context) {
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	status, ok := parseRangedOptionalInt(c.Query("status"), 0, 1)
	if !ok {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "status 仅支持 0/1"))
		return
	}
	result, err := a.endpoints.List(c.Request.Context(), pageNum, pageSize, c.Query("keyword"), status)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

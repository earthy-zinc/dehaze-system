package aidomain

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// AgentRepository AI 智能体、关联关系、版本快照与外部 A2A 端点的数据访问。
type AgentRepository struct {
	db *gorm.DB
}

func NewAgentRepository(db *gorm.DB) *AgentRepository {
	return &AgentRepository{db: db}
}

// ── Agent 主表 ────────────────────────────────────────────────

func (r *AgentRepository) Create(ctx context.Context, agent *model.SysAiAgent) error {
	return r.db.WithContext(ctx).Create(agent).Error
}

func (r *AgentRepository) GetByID(ctx context.Context, id int64) (*model.SysAiAgent, error) {
	var agent model.SysAiAgent
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&agent).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &agent, err
}

func (r *AgentRepository) GetByCode(ctx context.Context, code string) (*model.SysAiAgent, error) {
	var agent model.SysAiAgent
	err := r.db.WithContext(ctx).
		Where("agent_code = ? AND deleted = 0", code).First(&agent).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &agent, err
}

func (r *AgentRepository) GetByIDs(ctx context.Context, ids []int64) ([]model.SysAiAgent, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	var agents []model.SysAiAgent
	err := r.db.WithContext(ctx).Where("id IN ? AND deleted = 0", ids).Find(&agents).Error
	return agents, err
}

// ListAll 全部未删除 Agent（评测中心总览聚合用）。
func (r *AgentRepository) ListAll(ctx context.Context) ([]model.SysAiAgent, error) {
	var agents []model.SysAiAgent
	err := r.db.WithContext(ctx).Where("deleted = 0").
		Order("id ASC").Find(&agents).Error
	return agents, err
}

func (r *AgentRepository) Paginate(ctx context.Context, page, size int, keyword string, status *int, agentType string) ([]model.SysAiAgent, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiAgent{}).Where("deleted = 0")
	if keyword != "" {
		pattern := "%" + escapeLike(keyword) + "%"
		db = db.Where("(name LIKE ? ESCAPE '\\\\' OR agent_code LIKE ? ESCAPE '\\\\')", pattern, pattern)
	}
	if status != nil {
		db = db.Where("status = ?", *status)
	}
	switch agentType {
	case "subagent":
		db = db.Where("is_subagent = 1")
	case "team":
		db = db.Where("is_team = 1")
	case "agent":
		db = db.Where("is_subagent = 0 AND is_team = 0")
	}
	db = db.Order("sort_order ASC, id ASC")
	return countAndFind[model.SysAiAgent](db, page, size)
}

// ListEnabled 可选 Agent（启用且非子 Agent）。
func (r *AgentRepository) ListEnabled(ctx context.Context) ([]model.SysAiAgent, error) {
	var agents []model.SysAiAgent
	err := r.db.WithContext(ctx).
		Where("status = 1 AND is_subagent = 0 AND deleted = 0").
		Order("sort_order ASC, id ASC").Find(&agents).Error
	return agents, err
}

// UpdateFields 更新 Agent 可编辑态字段。
func (r *AgentRepository) UpdateFields(ctx context.Context, id int64, fields map[string]any) error {
	fields["update_time"] = time.Now()
	return r.db.WithContext(ctx).Model(&model.SysAiAgent{}).
		Where("id = ?", id).Updates(fields).Error
}

func (r *AgentRepository) SoftDeleteByIDs(ctx context.Context, ids []int64, updateBy int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Model(&model.SysAiAgent{}).
		Where("id IN ?", ids).
		Updates(map[string]any{
			"deleted":     gorm.Expr("id"),
			"update_time": time.Now(),
			"update_by":   updateBy,
		}).Error
}

// CountConversationReferences 统计使用该 agent_code 的未删会话数。
func (r *AgentRepository) CountConversationReferences(ctx context.Context, agentCode string) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).Table("sys_ai_conversation").
		Where("agent_code = ? AND deleted = 0", agentCode).
		Count(&count).Error
	return count, err
}

// CountSubagentReferences 统计把该 Agent 作为子 Agent 引用的关联数。
func (r *AgentRepository) CountSubagentReferences(ctx context.Context, agentID int64) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysAiAgentSubagent{}).
		Where("subagent_agent_id = ?", agentID).
		Count(&count).Error
	return count, err
}

// ── 关联关系 ──────────────────────────────────────────────────

func (r *AgentRepository) ListSkillNames(ctx context.Context, agentID int64) ([]string, error) {
	names := []string{}
	err := r.db.WithContext(ctx).Model(&model.SysAiAgentSkill{}).
		Where("agent_id = ?", agentID).
		Order("skill_name ASC").Pluck("skill_name", &names).Error
	return names, err
}

func (r *AgentRepository) ListMcpNamespaces(ctx context.Context, agentID int64) ([]string, error) {
	namespaces := []string{}
	err := r.db.WithContext(ctx).Model(&model.SysAiAgentMcp{}).
		Where("agent_id = ?", agentID).
		Order("mcp_namespace ASC").Pluck("mcp_namespace", &namespaces).Error
	return namespaces, err
}

func (r *AgentRepository) ListSubagents(ctx context.Context, agentID int64) ([]model.SysAiAgentSubagent, error) {
	var links []model.SysAiAgentSubagent
	err := r.db.WithContext(ctx).
		Where("parent_agent_id = ?", agentID).
		Order("priority ASC, subagent_agent_id ASC").Find(&links).Error
	return links, err
}

// SubAgentItem 子 Agent 关联（含被引用 Agent 的展示字段）。
type SubAgentItem struct {
	AgentID     int64  `gorm:"column:agent_id"`
	AgentName   string `gorm:"column:agent_name"`
	AgentCode   string `gorm:"column:agent_code"`
	Description string `gorm:"column:description"`
	EndpointID  *int64 `gorm:"column:endpoint_id"`
	Priority    int    `gorm:"column:priority"`
}

// ListSubagentItems 加载子 Agent 关联详情（一次 JOIN，避免 N+1）。
func (r *AgentRepository) ListSubagentItems(ctx context.Context, agentID int64) ([]SubAgentItem, error) {
	var items []SubAgentItem
	err := r.db.WithContext(ctx).Table("sys_ai_agent_subagent s").
		Select("s.subagent_agent_id AS agent_id, COALESCE(a.name, '') AS agent_name, COALESCE(a.agent_code, '') AS agent_code, COALESCE(a.description, '') AS description, s.endpoint_id, s.priority").
		Joins("LEFT JOIN sys_ai_agent a ON a.id = s.subagent_agent_id AND a.deleted = 0").
		Where("s.parent_agent_id = ?", agentID).
		Order("s.priority ASC, s.subagent_agent_id ASC").
		Scan(&items).Error
	return items, err
}

// ReplaceSkills 覆盖式更新 Agent-Skill 关联。
func (r *AgentRepository) ReplaceSkills(ctx context.Context, agentID int64, names []string) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err := tx.Where("agent_id = ?", agentID).Delete(&model.SysAiAgentSkill{}).Error; err != nil {
			return err
		}
		for _, name := range names {
			if err := tx.Create(&model.SysAiAgentSkill{AgentID: agentID, SkillName: name}).Error; err != nil {
				return err
			}
		}
		return nil
	})
}

// ReplaceMcpNamespaces 覆盖式更新 Agent-MCP 关联。
func (r *AgentRepository) ReplaceMcpNamespaces(ctx context.Context, agentID int64, namespaces []string) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err := tx.Where("agent_id = ?", agentID).Delete(&model.SysAiAgentMcp{}).Error; err != nil {
			return err
		}
		for _, ns := range namespaces {
			if err := tx.Create(&model.SysAiAgentMcp{AgentID: agentID, McpNamespace: ns}).Error; err != nil {
				return err
			}
		}
		return nil
	})
}

// ReplaceSubagents 覆盖式更新 Agent-Subagent 关联。
func (r *AgentRepository) ReplaceSubagents(ctx context.Context, agentID int64, items []model.SysAiAgentSubagent) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err := tx.Where("parent_agent_id = ?", agentID).Delete(&model.SysAiAgentSubagent{}).Error; err != nil {
			return err
		}
		for _, item := range items {
			item.ParentAgentID = agentID
			if err := tx.Create(&item).Error; err != nil {
				return err
			}
		}
		return nil
	})
}

// CountLinksByAgentIDs 批量统计关联数（skill/mcp/subagent）。
func (r *AgentRepository) CountLinksByAgentIDs(ctx context.Context, table, column string, agentIDs []int64) (map[int64]int64, error) {
	result := make(map[int64]int64)
	if len(agentIDs) == 0 {
		return result, nil
	}
	type row struct {
		AgentID int64 `gorm:"column:agent_id"`
		Total   int64 `gorm:"column:total"`
	}
	var rows []row
	err := r.db.WithContext(ctx).Table(table).
		Select("agent_id, COUNT(*) AS total").
		Where("agent_id IN ?", agentIDs).
		Group("agent_id").Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	for _, item := range rows {
		result[item.AgentID] = item.Total
	}
	return result, nil
}

// CountSubagentsByAgentIDs 批量统计子 Agent 关联数。
func (r *AgentRepository) CountSubagentsByAgentIDs(ctx context.Context, agentIDs []int64) (map[int64]int64, error) {
	result := make(map[int64]int64)
	if len(agentIDs) == 0 {
		return result, nil
	}
	type row struct {
		AgentID int64 `gorm:"column:agent_id"`
		Total   int64 `gorm:"column:total"`
	}
	var rows []row
	err := r.db.WithContext(ctx).Table("sys_ai_agent_subagent").
		Select("parent_agent_id AS agent_id, COUNT(*) AS total").
		Where("parent_agent_id IN ?", agentIDs).
		Group("parent_agent_id").Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	for _, item := range rows {
		result[item.AgentID] = item.Total
	}
	return result, nil
}

// ListExistingSkillNames 返回入参中真实存在的 Skill 名称（未删除）。
func (r *AgentRepository) ListExistingSkillNames(ctx context.Context, names []string) ([]string, error) {
	if len(names) == 0 {
		return nil, nil
	}
	var existing []string
	err := r.db.WithContext(ctx).Table("sys_ai_skill").
		Where("name IN ? AND deleted = 0", names).
		Pluck("name", &existing).Error
	return existing, err
}

// ListRegisteredNamespaces 返回入参中已注册的 MCP 命名空间。
func (r *AgentRepository) ListRegisteredNamespaces(ctx context.Context, namespaces []string) ([]string, error) {
	if len(namespaces) == 0 {
		return nil, nil
	}
	var registered []string
	err := r.db.WithContext(ctx).Table("sys_ai_mcp_namespace").
		Where("namespace IN ?", namespaces).
		Pluck("namespace", &registered).Error
	return registered, err
}

// ── 版本快照 ──────────────────────────────────────────────────

func (r *AgentRepository) NextVersionNo(ctx context.Context, agentID int64) (int, error) {
	var maxNo *int
	err := r.db.WithContext(ctx).Model(&model.SysAiAgentVersion{}).
		Select("MAX(version_no)").
		Where("agent_id = ?", agentID).
		Scan(&maxNo).Error
	if err != nil {
		return 0, err
	}
	if maxNo == nil {
		return 1, nil
	}
	return *maxNo + 1, nil
}

func (r *AgentRepository) CreateVersion(ctx context.Context, version *model.SysAiAgentVersion) error {
	return r.db.WithContext(ctx).Create(version).Error
}

func (r *AgentRepository) GetVersion(ctx context.Context, agentID int64, versionNo int) (*model.SysAiAgentVersion, error) {
	var version model.SysAiAgentVersion
	err := r.db.WithContext(ctx).
		Where("agent_id = ? AND version_no = ?", agentID, versionNo).
		First(&version).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &version, err
}

func (r *AgentRepository) GetLatestPublished(ctx context.Context, agentID int64) (*model.SysAiAgentVersion, error) {
	var version model.SysAiAgentVersion
	err := r.db.WithContext(ctx).
		Where("agent_id = ? AND status = 2", agentID).
		Order("version_no DESC").First(&version).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &version, err
}

// ListVersions 版本历史分页（不加载 snapshot 大字段）。
func (r *AgentRepository) ListVersions(ctx context.Context, agentID int64, offset, limit int) ([]model.SysAiAgentVersion, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiAgentVersion{}).Where("agent_id = ?", agentID)
	var total int64
	if err := db.Session(&gorm.Session{}).Count(&total).Error; err != nil {
		return nil, 0, err
	}
	var items []model.SysAiAgentVersion
	err := db.Select("id, agent_id, version_no, status, change_note, operator_id, create_time").
		Order("version_no DESC").Offset(offset).Limit(limit).Find(&items).Error
	return items, total, err
}

// DemotePublished 将既有已发布版本置为历史（草稿状态）。
func (r *AgentRepository) DemotePublished(ctx context.Context, agentID int64) error {
	return r.db.WithContext(ctx).Model(&model.SysAiAgentVersion{}).
		Where("agent_id = ? AND status = 2", agentID).
		Update("status", 1).Error
}

// LoadDictValues 读取字典类型下启用的 {name: value}（快照 resolved_config 与评测阈值用）。
func (r *AgentRepository) LoadDictValues(ctx context.Context, typeCode string) (map[string]string, error) {
	result := make(map[string]string)
	type row struct {
		Name  string `gorm:"column:name"`
		Value string `gorm:"column:value"`
	}
	var rows []row
	err := r.db.WithContext(ctx).Table("sys_dict").
		Select("name, value").
		Where("type_code = ? AND status = 1 AND deleted = 0", typeCode).
		Order("sort ASC").Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	for _, item := range rows {
		result[item.Name] = item.Value
	}
	return result, nil
}

// ── 外部 A2A 端点 ─────────────────────────────────────────────

func (r *AgentRepository) CreateEndpoint(ctx context.Context, endpoint *model.SysAiAgentEndpoint) error {
	return r.db.WithContext(ctx).Create(endpoint).Error
}

func (r *AgentRepository) GetEndpoint(ctx context.Context, id int64) (*model.SysAiAgentEndpoint, error) {
	var endpoint model.SysAiAgentEndpoint
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&endpoint).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &endpoint, err
}

// GetEndpointByBaseURL 按地址查端点（含软删行，用于复活唯一键占用行）。
func (r *AgentRepository) GetEndpointByBaseURL(ctx context.Context, baseURL string) (*model.SysAiAgentEndpoint, error) {
	var endpoint model.SysAiAgentEndpoint
	err := r.db.WithContext(ctx).Where("base_url = ?", baseURL).First(&endpoint).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &endpoint, err
}

func (r *AgentRepository) PaginateEndpoints(ctx context.Context, page, size int, keyword string, status *int) ([]model.SysAiAgentEndpoint, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiAgentEndpoint{}).Where("deleted = 0")
	if keyword != "" {
		pattern := "%" + escapeLike(keyword) + "%"
		db = db.Where("(name LIKE ? ESCAPE '\\\\' OR base_url LIKE ? ESCAPE '\\\\')", pattern, pattern)
	}
	if status != nil {
		db = db.Where("status = ?", *status)
	}
	db = db.Order("id DESC")
	return countAndFind[model.SysAiAgentEndpoint](db, page, size)
}

func (r *AgentRepository) UpdateEndpointFields(ctx context.Context, id int64, fields map[string]any) error {
	fields["update_time"] = time.Now()
	return r.db.WithContext(ctx).Model(&model.SysAiAgentEndpoint{}).
		Where("id = ?", id).Updates(fields).Error
}

func (r *AgentRepository) SoftDeleteEndpoints(ctx context.Context, ids []int64, updateBy int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Model(&model.SysAiAgentEndpoint{}).
		Where("id IN ?", ids).
		Updates(map[string]any{
			"deleted":     gorm.Expr("id"),
			"update_time": time.Now(),
			"update_by":   updateBy,
		}).Error
}

// ReviveEndpoint 复活软删行（唯一键 base_url 被软删行占用时）。
func (r *AgentRepository) ReviveEndpoint(ctx context.Context, id int64, fields map[string]any) error {
	fields["deleted"] = 0
	fields["update_time"] = time.Now()
	return r.db.WithContext(ctx).Model(&model.SysAiAgentEndpoint{}).
		Where("id = ?", id).Updates(fields).Error
}

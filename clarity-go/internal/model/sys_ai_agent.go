package model

import "time"

// SysAiAgent AI 智能体配置
type SysAiAgent struct {
	ID            int64      `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	AgentCode     string     `gorm:"column:agent_code" json:"agentCode"`
	Name          string     `gorm:"column:name" json:"name"`
	Description   string     `gorm:"column:description" json:"description"`
	SystemPrompt  *string    `gorm:"column:system_prompt" json:"systemPrompt"`
	ModelID       string     `gorm:"column:model_id" json:"modelId"`
	ReasoningMode string     `gorm:"column:reasoning_mode" json:"reasoningMode"`
	Config        string     `gorm:"column:config;type:json;default:null" json:"config"`
	IsSubagent    int        `gorm:"column:is_subagent" json:"isSubagent"`
	IsTeam        int        `gorm:"column:is_team" json:"isTeam"`
	IsExposed     int        `gorm:"column:is_exposed" json:"isExposed"`
	Permissions   string     `gorm:"column:permissions;type:json;default:null" json:"permissions"`
	Tags          string     `gorm:"column:tags;type:json;default:null" json:"tags"`
	SortOrder     int        `gorm:"column:sort_order" json:"sortOrder"`
	Status        int        `gorm:"column:status" json:"status"`
	Deleted       int64      `gorm:"column:deleted" json:"deleted"`
	CreateBy      *int64     `gorm:"column:create_by" json:"createBy"`
	UpdateBy      *int64     `gorm:"column:update_by" json:"updateBy"`
	CreateTime    time.Time  `gorm:"column:create_time;autoCreateTime" json:"createTime"`
	UpdateTime    *time.Time `gorm:"column:update_time;autoUpdateTime" json:"updateTime"`
}

func (SysAiAgent) TableName() string { return "sys_ai_agent" }

// SysAiAgentSkill Agent-Skill 关联
type SysAiAgentSkill struct {
	AgentID    int64     `gorm:"column:agent_id;primaryKey" json:"agentId"`
	SkillName  string    `gorm:"column:skill_name;primaryKey" json:"skillName"`
	CreateTime time.Time `gorm:"column:create_time;autoCreateTime" json:"createTime"`
}

func (SysAiAgentSkill) TableName() string { return "sys_ai_agent_skill" }

// SysAiAgentMcp Agent-MCP 命名空间关联
type SysAiAgentMcp struct {
	AgentID      int64     `gorm:"column:agent_id;primaryKey" json:"agentId"`
	McpNamespace string    `gorm:"column:mcp_namespace;primaryKey" json:"mcpNamespace"`
	CreateTime   time.Time `gorm:"column:create_time;autoCreateTime" json:"createTime"`
}

func (SysAiAgentMcp) TableName() string { return "sys_ai_agent_mcp" }

// SysAiAgentSubagent Agent-Subagent 关联（自引用）
type SysAiAgentSubagent struct {
	ParentAgentID   int64     `gorm:"column:parent_agent_id;primaryKey" json:"parentAgentId"`
	SubagentAgentID int64     `gorm:"column:subagent_agent_id;primaryKey" json:"subagentAgentId"`
	EndpointID      *int64    `gorm:"column:endpoint_id" json:"endpointId"`
	Priority        int       `gorm:"column:priority" json:"priority"`
	CreateTime      time.Time `gorm:"column:create_time;autoCreateTime" json:"createTime"`
}

func (SysAiAgentSubagent) TableName() string { return "sys_ai_agent_subagent" }

// SysAiAgentVersion Agent 配置版本快照（只追加）
type SysAiAgentVersion struct {
	ID        int64 `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	AgentID   int64 `gorm:"column:agent_id" json:"agentId"`
	VersionNo int   `gorm:"column:version_no" json:"versionNo"`
	// snapshot 列为 NOT NULL 且无数据库默认值：不能用 default:null（会因省略列报 1364），
	// 零值须由 GORM 兜底写入合法 JSON 对象
	Snapshot   string    `gorm:"column:snapshot;type:json;not null;default:'{}'" json:"snapshot"`
	Status     int       `gorm:"column:status" json:"status"`
	ChangeNote *string   `gorm:"column:change_note" json:"changeNote"`
	OperatorID *int64    `gorm:"column:operator_id" json:"operatorId"`
	CreateTime time.Time `gorm:"column:create_time;autoCreateTime" json:"createTime"`
}

func (SysAiAgentVersion) TableName() string { return "sys_ai_agent_version" }

// SysAiAgentEndpoint 外部 A2A 端点注册
type SysAiAgentEndpoint struct {
	ID           int64      `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	Name         string     `gorm:"column:name" json:"name"`
	AgentCardURL *string    `gorm:"column:agent_card_url" json:"agentCardUrl"`
	BaseURL      string     `gorm:"column:base_url" json:"baseUrl"`
	AuthType     string     `gorm:"column:auth_type" json:"authType"`
	Credential   *string    `gorm:"column:credential" json:"credential"`
	AgentCard    string     `gorm:"column:agent_card;type:json;default:null" json:"agentCard"`
	Status       int        `gorm:"column:status" json:"status"`
	Deleted      int64      `gorm:"column:deleted" json:"deleted"`
	CreateBy     *int64     `gorm:"column:create_by" json:"createBy"`
	UpdateBy     *int64     `gorm:"column:update_by" json:"updateBy"`
	CreateTime   time.Time  `gorm:"column:create_time;autoCreateTime" json:"createTime"`
	UpdateTime   *time.Time `gorm:"column:update_time;autoUpdateTime" json:"updateTime"`
}

func (SysAiAgentEndpoint) TableName() string { return "sys_ai_agent_endpoint" }

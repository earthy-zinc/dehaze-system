package model

import "encoding/json"

// SysAiMcpNamespace MCP Server 命名空间（工具分组）配置
type SysAiMcpNamespace struct {
	BaseModel
	ServerID  int64           `gorm:"column:server_id;type:bigint;not null;index:idx_server_id;comment:关联Server ID" json:"serverId"`
	Namespace string          `gorm:"column:namespace;type:varchar(128);not null;comment:命名空间标识" json:"namespace"`
	ToolNames json.RawMessage `gorm:"column:tool_names;type:json;comment:分组内工具名数组(JSON)" json:"toolNames"`
}

func (SysAiMcpNamespace) TableName() string {
	return "sys_ai_mcp_namespace"
}

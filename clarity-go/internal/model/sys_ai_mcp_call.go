package model

import (
	"encoding/json"
	"time"
)

// SysAiMcpCall 外部 MCP 工具调用审计（只追加，不逻辑删除）
type SysAiMcpCall struct {
	ID         int64           `gorm:"primaryKey;autoIncrement;column:id" json:"id"`
	UserID     *int64          `gorm:"column:user_id;type:bigint;comment:调用用户ID" json:"userId"`
	ServerID   int64           `gorm:"column:server_id;type:bigint;not null;comment:关联Server ID" json:"serverId"`
	ServerName *string         `gorm:"column:server_name;type:varchar(128);comment:Server名称(冗余快照)" json:"serverName"`
	ToolName   string          `gorm:"column:tool_name;type:varchar(128);not null;comment:被调用工具名" json:"toolName"`
	Request    json.RawMessage `gorm:"column:request;type:json;comment:调用载荷" json:"request"`
	Response   *string         `gorm:"column:response;type:text;comment:响应结果" json:"response"`
	Status     int8            `gorm:"column:status;type:tinyint;not null;default:0;comment:调用状态(0:失败;1:成功)" json:"status"`
	Result     string          `gorm:"column:result;type:varchar(16);not null;default:success;comment:调用结果(success/failure)" json:"result"`
	LatencyMs  *int            `gorm:"column:latency_ms;type:int;comment:调用耗时(毫秒)" json:"latencyMs"`
	CreateTime time.Time       `gorm:"column:create_time;type:datetime;not null;autoCreateTime" json:"createTime"`
}

func (SysAiMcpCall) TableName() string {
	return "sys_ai_mcp_call"
}

package model

import (
	"encoding/json"
	"time"
)

// SysAiMcpServer 外部 MCP Server 注册表
type SysAiMcpServer struct {
	BaseModel
	Name          string          `gorm:"column:name;type:varchar(128);not null;comment:Server名称(唯一)" json:"name"`
	Description   *string         `gorm:"column:description;type:varchar(512);comment:描述" json:"description"`
	ProtocolType  string          `gorm:"column:protocol_type;type:varchar(32);not null;default:streamable-http;comment:传输协议" json:"protocolType"`
	Endpoint      *string         `gorm:"column:endpoint;type:varchar(512);comment:端点URL" json:"endpoint"`
	AuthType      *string         `gorm:"column:auth_type;type:varchar(32);comment:鉴权方式" json:"authType"`
	Credentials   json.RawMessage `gorm:"column:credentials;type:json;comment:凭据密文(JSON)" json:"-"`
	Health        *string         `gorm:"column:health;type:varchar(16);comment:健康状态(online/offline)" json:"health"`
	LastCheckTime *time.Time      `gorm:"column:last_check_time;type:datetime;comment:最近一次健康探测时间" json:"lastCheckTime"`
	Status        int8            `gorm:"column:status;type:tinyint;not null;default:1;comment:状态(1:启用;0:禁用)" json:"status"`
	ToolCount     int             `gorm:"column:tool_count;type:int;not null;default:0;comment:工具数量" json:"toolCount"`
	Deleted       int64           `gorm:"column:deleted;type:bigint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysAiMcpServer) TableName() string {
	return "sys_ai_mcp_server"
}

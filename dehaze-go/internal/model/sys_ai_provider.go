package model

import "encoding/json"

// SysAiProvider AI模型供应商配置表
type SysAiProvider struct {
	BaseModel
	ProviderCode        string          `gorm:"column:provider_code;type:varchar(32);not null;comment:供应商编码" json:"providerCode"`
	DisplayName         string          `gorm:"column:display_name;type:varchar(128);not null;comment:显示名称" json:"displayName"`
	ApiBaseUrl          string          `gorm:"column:api_base_url;type:varchar(512);not null;comment:API基础地址" json:"apiBaseUrl"`
	ProtocolType        string          `gorm:"column:protocol_type;type:varchar(32);not null;default:openai_compat;comment:协议类型" json:"protocolType"`
	AuthType            string          `gorm:"column:auth_type;type:varchar(32);not null;default:bearer;comment:认证方式" json:"authType"`
	DefaultHeaders      json.RawMessage `gorm:"column:default_headers;type:json;comment:默认请求头" json:"defaultHeaders"`
	SortOrder           int             `gorm:"column:sort_order;type:int;not null;default:0;comment:排序序号" json:"sortOrder"`
	HealthCheckEnabled  int8            `gorm:"column:health_check_enabled;type:tinyint;not null;default:1;comment:健康检查开关" json:"healthCheckEnabled"`
	UserIdentityForward json.RawMessage `gorm:"column:user_identity_forward;type:json;comment:用户身份透传配置" json:"userIdentityForward"`
	Remark              *string         `gorm:"column:remark;type:varchar(512);comment:运维备注" json:"remark"`
	Status              int8            `gorm:"column:status;type:tinyint;not null;default:1;comment:状态(1:启用;0:禁用)" json:"status"`
	Deleted             int64           `gorm:"column:deleted;type:bigint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysAiProvider) TableName() string {
	return "sys_ai_provider"
}

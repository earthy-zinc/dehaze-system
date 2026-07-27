package model

// SysMessageTemplate 消息模板表
type SysMessageTemplate struct {
	BaseModel
	Code            string `gorm:"column:code;type:varchar(64);uniqueIndex:uk_code;not null;comment:模板编码" json:"code"`
	Name            string `gorm:"column:name;type:varchar(128);default:'';comment:模板名称" json:"name"`
	Type            string `gorm:"column:type;type:varchar(32);not null;comment:消息类型" json:"type"`
	TitleTemplate   string `gorm:"column:title_template;type:varchar(255);default:'';comment:标题模板" json:"titleTemplate"`
	ContentTemplate string `gorm:"column:content_template;type:text;not null;comment:正文模板" json:"contentTemplate"`
	Priority        int8   `gorm:"column:priority;type:tinyint;not null;default:2;comment:默认优先级" json:"priority"`
	Channels        string `gorm:"column:channels;type:json;comment:默认推送渠道JSON" json:"channels"`
	Variables       string `gorm:"column:variables;type:json;comment:变量定义JSON" json:"variables"`
	Status          int8   `gorm:"column:status;type:tinyint;not null;default:1;comment:状态(1:启用;0:禁用)" json:"status"`
	Deleted         int8   `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysMessageTemplate) TableName() string {
	return "sys_message_template"
}

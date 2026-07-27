package model

import "time"

// SysMessage 消息表
type SysMessage struct {
	BaseModel
	Type        string     `gorm:"column:type;type:varchar(32);not null;comment:消息类型" json:"type"`
	Title       string     `gorm:"column:title;type:varchar(255);default:'';comment:消息标题" json:"title"`
	Content     string     `gorm:"column:content;type:text;not null;comment:消息正文" json:"content"`
	SenderType  int8       `gorm:"column:sender_type;type:tinyint;not null;default:1;comment:发送者类型(1:系统;2:管理员)" json:"senderType"`
	RecipientID int64      `gorm:"column:recipient_id;type:bigint;not null;comment:接收人ID" json:"recipientId"`
	BizModule   string     `gorm:"column:biz_module;type:varchar(32);comment:业务模块" json:"bizModule"`
	BizID       string     `gorm:"column:biz_id;type:varchar(64);comment:业务ID" json:"bizId"`
	Priority    int8       `gorm:"column:priority;type:tinyint;not null;default:2;comment:优先级" json:"priority"`
	JumpURL     string     `gorm:"column:jump_url;type:varchar(255);comment:跳转链接" json:"jumpUrl"`
	Extra       string     `gorm:"column:extra;type:json;comment:扩展数据JSON" json:"extra"`
	ReadStatus  int8       `gorm:"column:read_status;type:tinyint;not null;default:0;comment:已读状态(0:未读;1:已读)" json:"readStatus"`
	ReadTime    *time.Time `gorm:"column:read_time;type:datetime;comment:已读时间" json:"readTime"`
	Deleted     int8       `gorm:"column:deleted;type:tinyint;not null;default:0;comment:用户删除标识" json:"deleted"`
	ExpiresAt   *time.Time `gorm:"column:expires_at;type:datetime;comment:过期时间" json:"expiresAt"`
}

func (SysMessage) TableName() string {
	return "sys_message"
}

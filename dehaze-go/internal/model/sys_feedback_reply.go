package model

import "time"

type SysFeedbackReply struct {
	ID          int64     `gorm:"primaryKey;autoIncrement;column:id" json:"id"`
	FeedbackID  int64     `gorm:"column:feedback_id;type:bigint;not null;index:idx_feedback_id_create_time,priority:1;comment:反馈ID" json:"feedbackId"`
	ReplierID   int64     `gorm:"column:replier_id;type:bigint;not null;index:idx_replier_id;comment:回复人ID" json:"replierId"`
	ReplierType int8      `gorm:"column:replier_type;type:tinyint;not null;comment:回复人类型(1:用户;2:管理员)" json:"replierType"`
	Content     string    `gorm:"column:content;type:varchar(2000);not null;comment:回复内容" json:"content"`
	ReplyType   string    `gorm:"column:reply_type;type:varchar(32);comment:回复类型(info:信息补充;resolved:已解决;unsupported:暂不支持;dev_transfer:转开发)" json:"replyType"`
	Attachments string    `gorm:"column:attachments;type:json;comment:附件URL（JSON数组）" json:"attachments"`
	Deleted     int8      `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"-"`
	CreateTime  time.Time `gorm:"column:create_time;type:datetime;default:CURRENT_TIMESTAMP;index:idx_feedback_id_create_time,priority:2;comment:创建时间" json:"createTime"`
	UpdateTime  time.Time `gorm:"column:update_time;type:datetime;default:CURRENT_TIMESTAMP;comment:更新时间" json:"updateTime"`
	CreateBy    *int64    `gorm:"column:create_by;comment:创建人ID" json:"createBy"`
	UpdateBy    *int64    `gorm:"column:update_by;comment:修改人ID" json:"updateBy"`
}

func (SysFeedbackReply) TableName() string {
	return "sys_feedback_reply"
}

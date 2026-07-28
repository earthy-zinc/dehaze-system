package model

import "time"

type SysFeedback struct {
	BaseModel
	UserID        int64      `gorm:"column:user_id;type:bigint;not null;index:idx_user_id;comment:提交用户ID" json:"userId"`
	FeedbackType  string     `gorm:"column:feedback_type;type:varchar(32);not null;index:idx_feedback_type;comment:反馈类型" json:"feedbackType"`
	Title         string     `gorm:"column:title;type:varchar(50);not null;comment:反馈标题" json:"title"`
	Content       string     `gorm:"column:content;type:varchar(1000);not null;comment:反馈内容" json:"content"`
	Contact       string     `gorm:"column:contact;type:varchar(64);comment:联系方式" json:"contact"`
	Images        string     `gorm:"column:images;type:json;comment:截图URL（JSON数组）" json:"images"`
	RelatedModule string     `gorm:"column:related_module;type:varchar(32);comment:相关模块" json:"relatedModule"`
	Status        int8       `gorm:"column:status;type:tinyint;not null;default:1;index:idx_status;index:idx_priority_status,priority:2;comment:状态(1:待处理;2:处理中;3:已回复;4:已关闭)" json:"status"`
	Priority      int8       `gorm:"column:priority;type:tinyint;not null;default:1;index:idx_priority_status,priority:1;comment:优先级(1:普通;2:紧急;3:高优)" json:"priority"`
	AssigneeID    *int64     `gorm:"column:assignee_id;type:bigint;index:idx_assignee_id;comment:处理人ID" json:"assigneeId"`
	AssignedTime  *time.Time `gorm:"column:assigned_time;type:datetime;comment:分配时间" json:"assignedTime"`
	Tags          string     `gorm:"column:tags;type:json;comment:反馈标签（JSON数组）" json:"tags"`
	CloseReason   string     `gorm:"column:close_reason;type:varchar(256);comment:关闭原因" json:"closeReason"`
	Deleted       int8       `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysFeedback) TableName() string {
	return "sys_feedback"
}

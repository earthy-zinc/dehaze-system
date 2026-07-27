package model

import "time"

// SysAnnouncement 系统公告表
type SysAnnouncement struct {
	BaseModel
	Title        string     `gorm:"column:title;type:varchar(255);default:'';comment:公告标题" json:"title"`
	Content      string     `gorm:"column:content;type:text;not null;comment:公告内容" json:"content"`
	Type         string     `gorm:"column:type;type:varchar(32);not null;default:operation;comment:公告类型" json:"type"`
	Importance   int8       `gorm:"column:importance;type:tinyint;not null;default:1;comment:重要级别(1:普通;2:重要)" json:"importance"`
	TargetScope  string     `gorm:"column:target_scope;type:varchar(32);not null;default:all;comment:发送范围" json:"targetScope"`
	TargetParams string     `gorm:"column:target_params;type:json;comment:范围参数JSON" json:"targetParams"`
	Status       int8       `gorm:"column:status;type:tinyint;not null;default:1;comment:公告状态(1:草稿;2:待发送;3:已发送;4:已取消)" json:"status"`
	SendTime     *time.Time `gorm:"column:send_time;type:datetime;comment:发送时间" json:"sendTime"`
	ExpireTime   *time.Time `gorm:"column:expire_time;type:datetime;comment:过期时间" json:"expireTime"`
	SentCount    int        `gorm:"column:sent_count;type:int;default:0;comment:已发送人数" json:"sentCount"`
	Deleted      int8       `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysAnnouncement) TableName() string {
	return "sys_announcement"
}

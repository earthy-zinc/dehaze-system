package model

import "time"

type SysRating struct {
	BaseModel
	UserID       int64      `gorm:"column:user_id;type:bigint;not null;index:idx_user_id;comment:评价用户ID" json:"userId"`
	PredLogID    int64      `gorm:"column:pred_log_id;type:bigint;not null;uniqueIndex:uk_pred_log_id;comment:关联处理日志ID" json:"predLogId"`
	AlgorithmID  int64      `gorm:"column:algorithm_id;type:bigint;not null;index:idx_algorithm_id;comment:算法ID" json:"algorithmId"`
	Rating       int8       `gorm:"column:rating;type:tinyint;not null;index:idx_rating;comment:评分(1-5星)" json:"rating"`
	Comment      string     `gorm:"column:comment;type:varchar(500);comment:评价文字" json:"comment"`
	Tags         string     `gorm:"column:tags;type:json;comment:评价标签（JSON数组）" json:"tags"`
	ImageUrls    string     `gorm:"column:image_urls;type:json;comment:截图URL（JSON数组）" json:"imageUrls"`
	IsAnonymous  int8       `gorm:"column:is_anonymous;type:tinyint;not null;default:0;comment:是否匿名(0:否;1:是)" json:"isAnonymous"`
	IsHidden     int8       `gorm:"column:is_hidden;type:tinyint;not null;default:0;comment:是否隐藏(0:否;1:是)" json:"isHidden"`
	AdminReply   string     `gorm:"column:admin_reply;type:varchar(2000);comment:管理员回复内容" json:"adminReply"`
	ReplyTime    *time.Time `gorm:"column:reply_time;type:datetime;comment:管理员回复时间" json:"replyTime"`
	Deleted      int8       `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysRating) TableName() string {
	return "sys_rating"
}

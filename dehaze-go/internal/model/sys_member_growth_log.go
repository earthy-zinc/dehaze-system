package model

import "time"

type SysMemberGrowthLog struct {
	ID          int64     `gorm:"primaryKey;autoIncrement;column:id" json:"id"`
	UserID      int64     `gorm:"column:user_id;type:bigint;not null;index:idx_user_id_create_time,priority:1;comment:用户ID" json:"userId"`
	ChangeType  string    `gorm:"column:change_type;type:varchar(32);not null;index:idx_change_type;comment:变动类型" json:"changeType"`
	ChangeValue int       `gorm:"column:change_value;type:int;not null;comment:变动值" json:"changeValue"`
	Balance     int64     `gorm:"column:balance;type:bigint;not null;comment:变动后余额" json:"balance"`
	RelatedID   string    `gorm:"column:related_id;type:varchar(64);index:idx_related_id;comment:关联业务ID" json:"relatedId"`
	Reason      string    `gorm:"column:reason;type:varchar(256);comment:变动原因" json:"reason"`
	OperatorID  *int64    `gorm:"column:operator_id;type:bigint;comment:操作人ID" json:"operatorId"`
	Deleted     int8      `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"-"`
	CreateTime  time.Time `gorm:"column:create_time;type:datetime;default:CURRENT_TIMESTAMP;index:idx_user_id_create_time,priority:2;comment:创建时间" json:"createTime"`
	UpdateTime  time.Time `gorm:"column:update_time;type:datetime;default:CURRENT_TIMESTAMP;comment:更新时间" json:"updateTime"`
	CreateBy    *int64    `gorm:"column:create_by;comment:创建人ID" json:"createBy"`
	UpdateBy    *int64    `gorm:"column:update_by;comment:修改人ID" json:"updateBy"`
}

func (SysMemberGrowthLog) TableName() string {
	return "sys_member_growth_log"
}

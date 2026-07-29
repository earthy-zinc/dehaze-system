package model

import (
	"database/sql"
	"time"
)

type SysPromotion struct {
	BaseModel
	Name            string         `gorm:"column:name;type:varchar(64);not null;comment:活动名称" json:"name"`
	Type            string         `gorm:"column:type;type:varchar(32);not null;index:idx_type;comment:活动类型" json:"type"`
	Description     string         `gorm:"column:description;type:varchar(256);comment:活动描述" json:"description"`
	StartTime       time.Time      `gorm:"column:start_time;type:datetime;not null;index:idx_time_range,priority:1;comment:开始时间" json:"startTime"`
	EndTime         time.Time      `gorm:"column:end_time;type:datetime;not null;index:idx_time_range,priority:2;comment:结束时间" json:"endTime"`
	ActivityRules   sql.NullString `gorm:"column:activity_rules;type:json;comment:活动规则" json:"activityRules"`
	ApplicableScope sql.NullString `gorm:"column:applicable_scope;type:json;comment:适用套餐ID列表" json:"applicableScope"`
	NewUserOnly     int8           `gorm:"column:new_user_only;type:tinyint;not null;default:0;comment:是否新用户专享" json:"newUserOnly"`
	Status          int8           `gorm:"column:status;type:tinyint;not null;default:1;index:idx_status;comment:状态" json:"status"`
	Deleted         int8           `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysPromotion) TableName() string {
	return "sys_promotion"
}

type SysPromotionPackage struct {
	ID            int64     `gorm:"primaryKey;autoIncrement;column:id" json:"id"`
	PromotionID   int64     `gorm:"column:promotion_id;type:bigint;not null;uniqueIndex:uk_promotion_package,priority:1;comment:促销活动ID" json:"promotionId"`
	PackageID     int64     `gorm:"column:package_id;type:bigint;not null;uniqueIndex:uk_promotion_package,priority:2;index:idx_package_id;comment:套餐ID" json:"packageId"`
	DiscountType  string    `gorm:"column:discount_type;type:varchar(16);not null;comment:折扣类型" json:"discountType"`
	DiscountValue int64     `gorm:"column:discount_value;type:bigint;not null;default:0;comment:折扣值" json:"discountValue"`
	CreateTime    time.Time `gorm:"column:create_time;type:datetime;default:CURRENT_TIMESTAMP;comment:创建时间" json:"createTime"`
	UpdateTime    time.Time `gorm:"column:update_time;type:datetime;default:CURRENT_TIMESTAMP;comment:更新时间" json:"updateTime"`
	CreateBy      int64     `gorm:"column:create_by;type:bigint;comment:创建人ID" json:"createBy"`
	UpdateBy      int64     `gorm:"column:update_by;type:bigint;comment:更新人ID" json:"updateBy"`
}

func (SysPromotionPackage) TableName() string {
	return "sys_promotion_package"
}

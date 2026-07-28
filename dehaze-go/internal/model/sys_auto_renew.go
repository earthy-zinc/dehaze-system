package model

import "time"

type SysAutoRenew struct {
	BaseModel
	UserID          int64      `gorm:"column:user_id;type:bigint;not null;uniqueIndex:uk_user_package,priority:1;comment:用户ID" json:"userId"`
	PackageID       int64      `gorm:"column:package_id;type:bigint;not null;uniqueIndex:uk_user_package,priority:2;comment:套餐ID" json:"packageId"`
	PayMethod       string     `gorm:"column:pay_method;type:varchar(16);not null;comment:支付方式" json:"payMethod"`
	Status          int8       `gorm:"column:status;type:tinyint;not null;default:1;index:idx_status_renew_time,priority:1;comment:状态" json:"status"`
	NextRenewTime   *time.Time `gorm:"column:next_renew_time;type:datetime;index:idx_status_renew_time,priority:2;comment:下次扣款时间" json:"nextRenewTime"`
	FailCount       int        `gorm:"column:fail_count;type:int;not null;default:0;comment:连续失败次数" json:"failCount"`
	LastRenewOrderID *int64    `gorm:"column:last_renew_order_id;type:bigint;comment:上次续费订单ID" json:"lastRenewOrderId"`
	CloseReason     string     `gorm:"column:close_reason;type:varchar(256);comment:关闭原因" json:"closeReason"`
	Deleted         int8       `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysAutoRenew) TableName() string {
	return "sys_auto_renew"
}

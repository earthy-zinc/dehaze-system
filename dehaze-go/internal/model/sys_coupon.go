package model

import "time"

type SysCoupon struct {
	BaseModel
	Name            string `gorm:"column:name;type:varchar(64);not null;comment:优惠券名称" json:"name"`
	Type            string `gorm:"column:type;type:varchar(32);not null;index:idx_type;comment:类型" json:"type"`
	FaceValue       int64  `gorm:"column:face_value;type:bigint;not null;default:0;comment:面值" json:"faceValue"`
	Threshold       *int64 `gorm:"column:threshold;type:bigint;comment:使用门槛" json:"threshold"`
	ValidType       string `gorm:"column:valid_type;type:varchar(16);not null;comment:有效期类型" json:"validType"`
	ValidStart      *time.Time `gorm:"column:valid_start;type:datetime;comment:有效期开始" json:"validStart"`
	ValidEnd        *time.Time `gorm:"column:valid_end;type:datetime;comment:有效期结束" json:"validEnd"`
	ValidDays       *int   `gorm:"column:valid_days;type:int;comment:领取后有效天数" json:"validDays"`
	TotalQty        int    `gorm:"column:total_qty;type:int;not null;default:-1;comment:发放总量" json:"totalQty"`
	IssuedQty       int    `gorm:"column:issued_qty;type:int;not null;default:0;comment:已发放数量" json:"issuedQty"`
	UsedQty         int    `gorm:"column:used_qty;type:int;not null;default:0;comment:已使用数量" json:"usedQty"`
	PerUserLimit    int    `gorm:"column:per_user_limit;type:int;not null;default:1;comment:每人限领" json:"perUserLimit"`
	ApplicableScope string `gorm:"column:applicable_scope;type:json;comment:适用套餐ID列表" json:"applicableScope"`
	Status          int8   `gorm:"column:status;type:tinyint;not null;default:1;index:idx_status;comment:状态" json:"status"`
	Deleted         int8   `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysCoupon) TableName() string {
	return "sys_coupon"
}

type SysUserCoupon struct {
	BaseModel
	UserID      int64      `gorm:"column:user_id;type:bigint;not null;index:idx_user_id_status;comment:用户ID" json:"userId"`
	CouponID    int64      `gorm:"column:coupon_id;type:bigint;not null;index:idx_coupon_id;comment:优惠券模板ID" json:"couponId"`
	Status      int8       `gorm:"column:status;type:tinyint;not null;default:1;index:idx_user_id_status;comment:状态" json:"status"`
	ReceiveTime time.Time  `gorm:"column:receive_time;type:datetime;not null;default:CURRENT_TIMESTAMP;comment:领取时间" json:"receiveTime"`
	ExpireTime  *time.Time `gorm:"column:expire_time;type:datetime;index:idx_expire_time;comment:过期时间" json:"expireTime"`
	UsedTime    *time.Time `gorm:"column:used_time;type:datetime;comment:使用时间" json:"usedTime"`
	UsedOrderID *int64     `gorm:"column:used_order_id;type:bigint;index:idx_used_order_id;comment:使用的订单ID" json:"usedOrderId"`
	Deleted     int8       `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysUserCoupon) TableName() string {
	return "sys_user_coupon"
}

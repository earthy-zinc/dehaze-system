package model

import "time"

type SysOrder struct {
	BaseModel
	OrderNo          string     `gorm:"column:order_no;type:varchar(32);not null;uniqueIndex:uk_order_no;comment:订单号" json:"orderNo"`
	UserID           int64      `gorm:"column:user_id;type:bigint;not null;index:idx_user_id_status;comment:用户ID" json:"userId"`
	PackageID        int64      `gorm:"column:package_id;type:bigint;not null;comment:套餐ID" json:"packageId"`
	PackageName      string     `gorm:"column:package_name;type:varchar(32);not null;comment:套餐名称" json:"packageName"`
	PackageLevel     string     `gorm:"column:package_level;type:varchar(16);not null;comment:套餐会员等级" json:"packageLevel"`
	PeriodDays       int        `gorm:"column:period_days;type:int;not null;comment:有效期天数" json:"periodDays"`
	OriginalPrice    int64      `gorm:"column:original_price;type:bigint;not null;comment:原价" json:"originalPrice"`
	DiscountAmount   int64      `gorm:"column:discount_amount;type:bigint;not null;default:0;comment:促销折扣" json:"discountAmount"`
	CouponID         *int64     `gorm:"column:coupon_id;type:bigint;index:idx_coupon_id;comment:用户优惠券实例ID" json:"couponId"`
	CouponAmount     int64      `gorm:"column:coupon_amount;type:bigint;not null;default:0;comment:优惠券抵扣" json:"couponAmount"`
	PayableAmount    int64      `gorm:"column:payable_amount;type:bigint;not null;comment:应付金额" json:"payableAmount"`
	PaidAmount       int64      `gorm:"column:paid_amount;type:bigint;not null;default:0;comment:实付金额" json:"paidAmount"`
	PayMethod        *string    `gorm:"column:pay_method;type:varchar(16);comment:支付方式" json:"payMethod"`
	Status           int8       `gorm:"column:status;type:tinyint;not null;default:1;index:idx_user_id_status;index:idx_status;comment:订单状态" json:"status"`
	ExpireTime       time.Time  `gorm:"column:expire_time;type:datetime;not null;index:idx_expire_time;comment:支付超时时间" json:"expireTime"`
	EffectiveTime    *time.Time `gorm:"column:effective_time;type:datetime;comment:权益生效时间" json:"effectiveTime"`
	PackageExpireTime *time.Time `gorm:"column:package_expire_time;type:datetime;index:idx_package_expire_time;comment:套餐到期时间" json:"packageExpireTime"`
	PaidTime         *time.Time `gorm:"column:paid_time;type:datetime;comment:支付成功时间" json:"paidTime"`
	CancelReason     string     `gorm:"column:cancel_reason;type:varchar(256);comment:取消原因" json:"cancelReason"`
	IsAutoRenew      int8       `gorm:"column:is_auto_renew;type:tinyint;not null;default:0;comment:是否自动续费订单" json:"isAutoRenew"`
	Deleted          int8       `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysOrder) TableName() string {
	return "sys_order"
}

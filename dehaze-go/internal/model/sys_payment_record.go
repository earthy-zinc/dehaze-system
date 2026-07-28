package model

import "time"

type SysPaymentRecord struct {
	ID              int64      `gorm:"primaryKey;autoIncrement;column:id" json:"id"`
	OrderID         int64      `gorm:"column:order_id;type:bigint;not null;index:idx_order_id;comment:订单ID" json:"orderId"`
	UserID          int64      `gorm:"column:user_id;type:bigint;not null;index:idx_user_id;comment:用户ID" json:"userId"`
	PaymentNo       string     `gorm:"column:payment_no;type:varchar(64);not null;uniqueIndex:uk_payment_no;comment:支付渠道流水号" json:"paymentNo"`
	Channel         string     `gorm:"column:channel;type:varchar(16);not null;comment:支付渠道" json:"channel"`
	Amount          int64      `gorm:"column:amount;type:bigint;not null;comment:支付金额" json:"amount"`
	Status          int8       `gorm:"column:status;type:tinyint;not null;default:1;index:idx_status;comment:支付状态" json:"status"`
	CallbackTime    *time.Time `gorm:"column:callback_time;type:datetime;comment:回调到达时间" json:"callbackTime"`
	CallbackContent string     `gorm:"column:callback_content;type:text;comment:渠道回调原始报文" json:"callbackContent"`
	ErrorMessage    string     `gorm:"column:error_message;type:varchar(512);comment:错误信息" json:"errorMessage"`
	CreateTime      time.Time  `gorm:"column:create_time;type:datetime;not null;default:CURRENT_TIMESTAMP;comment:创建时间" json:"createTime"`
	UpdateTime      time.Time  `gorm:"column:update_time;type:datetime;not null;default:CURRENT_TIMESTAMP;comment:更新时间" json:"updateTime"`
}

func (SysPaymentRecord) TableName() string {
	return "sys_payment_record"
}

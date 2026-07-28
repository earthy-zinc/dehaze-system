package model

import "time"

type SysRefundRecord struct {
	BaseModel
	RefundNo        string     `gorm:"column:refund_no;type:varchar(32);not null;uniqueIndex:uk_refund_no;comment:退款单号" json:"refundNo"`
	OrderID         int64      `gorm:"column:order_id;type:bigint;not null;uniqueIndex:uk_order_id;comment:订单ID" json:"orderId"`
	UserID          int64      `gorm:"column:user_id;type:bigint;not null;index:idx_user_id;comment:用户ID" json:"userId"`
	RefundAmount    int64      `gorm:"column:refund_amount;type:bigint;not null;comment:退款金额" json:"refundAmount"`
	Reason          string     `gorm:"column:reason;type:varchar(256);not null;comment:退款原因" json:"reason"`
	UsedQuota       int        `gorm:"column:used_quota;type:int;not null;default:0;comment:申请时已用权益次数" json:"usedQuota"`
	Status          int8       `gorm:"column:status;type:tinyint;not null;default:1;index:idx_status;comment:退款状态" json:"status"`
	Channel         *string    `gorm:"column:channel;type:varchar(16);comment:退款渠道" json:"channel"`
	ChannelRefundNo string     `gorm:"column:channel_refund_no;type:varchar(64);comment:渠道退款流水号" json:"channelRefundNo"`
	ApplyTime       time.Time  `gorm:"column:apply_time;type:datetime;not null;default:CURRENT_TIMESTAMP;comment:申请时间" json:"applyTime"`
	AuditTime       *time.Time `gorm:"column:audit_time;type:datetime;comment:审核时间" json:"auditTime"`
	AuditorID       *int64     `gorm:"column:auditor_id;type:bigint;index:idx_auditor_id;comment:审核人ID" json:"auditorId"`
	AuditRemark     string     `gorm:"column:audit_remark;type:varchar(256);comment:审核备注" json:"auditRemark"`
	RefundTime      *time.Time `gorm:"column:refund_time;type:datetime;comment:退款完成时间" json:"refundTime"`
	ErrorMessage    string     `gorm:"column:error_message;type:varchar(512);comment:错误信息" json:"errorMessage"`
	Deleted         int8       `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysRefundRecord) TableName() string {
	return "sys_refund_record"
}

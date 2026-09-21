package model

import "time"

// SysBalance 平台人民币余额账户（充值/支付/退款回充），与 AI 计费积分账户职责分离
type SysBalance struct {
	BaseModel
	UserID        int64 `gorm:"column:user_id;type:bigint;not null;uniqueIndex:uk_user_id;comment:用户ID" json:"userId"`
	Balance       int64 `gorm:"column:balance;type:bigint;not null;default:0;comment:可用余额(分)" json:"balance"`
	FrozenBalance int64 `gorm:"column:frozen_balance;type:bigint;not null;default:0;comment:冻结余额(分)" json:"frozenBalance"`
	Version       int   `gorm:"column:version;type:int;not null;default:0;comment:乐观锁版本号" json:"version"`
	Deleted       int64 `gorm:"column:deleted;type:bigint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysBalance) TableName() string {
	return "sys_balance"
}

// SysBalanceLog 余额流水（资金审计追溯）
type SysBalanceLog struct {
	BaseModel
	UserID       int64  `gorm:"column:user_id;type:bigint;not null;index:idx_user_id;comment:用户ID" json:"userId"`
	ChangeType   string `gorm:"column:change_type;type:varchar(16);not null;comment:变动类型(recharge:充值;consume:消费;refund:退款退回;freeze:冻结;unfreeze:解冻)" json:"changeType"`
	Amount       int64  `gorm:"column:amount;type:bigint;not null;comment:变动金额(正数增加;负数扣减)" json:"amount"`
	BalanceAfter int64  `gorm:"column:balance_after;type:bigint;not null;comment:变动后可用余额" json:"balanceAfter"`
	RelatedID    *int64 `gorm:"column:related_id;type:bigint;comment:关联业务记录ID" json:"relatedId"`
	Deleted      int64  `gorm:"column:deleted;type:bigint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysBalanceLog) TableName() string {
	return "sys_balance_log"
}

// SysBalanceRefund 平台余额退款记录
type SysBalanceRefund struct {
	BaseModel
	RefundNo        string     `gorm:"column:refund_no;type:varchar(32);not null;uniqueIndex:uk_refund_no;comment:退款单号" json:"refundNo"`
	UserID          int64      `gorm:"column:user_id;type:bigint;not null;index:idx_user_id;comment:用户ID" json:"userId"`
	Amount          int64      `gorm:"column:amount;type:bigint;not null;comment:退款金额(分)" json:"amount"`
	Status          int8       `gorm:"column:status;type:tinyint;not null;default:1;index:idx_status;comment:退款状态(1:待审核;2:已退款;3:退款失败)" json:"status"`
	Channel         *string    `gorm:"column:channel;type:varchar(16);comment:原路退回渠道" json:"channel"`
	ChannelRefundNo string     `gorm:"column:channel_refund_no;type:varchar(64);comment:渠道退款流水号" json:"channelRefundNo"`
	ApplyTime       time.Time  `gorm:"column:apply_time;type:datetime;not null;default:CURRENT_TIMESTAMP;comment:申请时间" json:"applyTime"`
	AuditTime       *time.Time `gorm:"column:audit_time;type:datetime;comment:审核时间" json:"auditTime"`
	AuditorID       *int64     `gorm:"column:auditor_id;type:bigint;comment:审核人ID" json:"auditorId"`
	AuditRemark     string     `gorm:"column:audit_remark;type:varchar(256);comment:审核备注" json:"auditRemark"`
	RefundTime      *time.Time `gorm:"column:refund_time;type:datetime;comment:退款完成时间" json:"refundTime"`
	ErrorMessage    string     `gorm:"column:error_message;type:varchar(512);comment:错误信息" json:"errorMessage"`
	Deleted         int64      `gorm:"column:deleted;type:bigint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysBalanceRefund) TableName() string {
	return "sys_balance_refund"
}

// SysRecharge 余额充值单
type SysRecharge struct {
	BaseModel
	RechargeNo       string     `gorm:"column:recharge_no;type:varchar(32);not null;uniqueIndex:uk_recharge_no;comment:充值单号" json:"rechargeNo"`
	UserID           int64      `gorm:"column:user_id;type:bigint;not null;index:idx_user_id;comment:用户ID" json:"userId"`
	Amount           int64      `gorm:"column:amount;type:bigint;not null;comment:充值金额(分)" json:"amount"`
	PayMethod        string     `gorm:"column:pay_method;type:varchar(16);not null;comment:支付方式(wechat:微信;alipay:支付宝)" json:"payMethod"`
	Status           int8       `gorm:"column:status;type:tinyint;not null;default:1;comment:充值状态(1:待支付;2:已支付;3:已关闭)" json:"status"`
	ChannelPaymentNo *string    `gorm:"column:channel_payment_no;type:varchar(64);comment:渠道支付流水号" json:"channelPaymentNo"`
	PayTime          *time.Time `gorm:"column:pay_time;type:datetime;comment:支付成功时间" json:"payTime"`
	Deleted          int64      `gorm:"column:deleted;type:bigint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysRecharge) TableName() string {
	return "sys_recharge"
}

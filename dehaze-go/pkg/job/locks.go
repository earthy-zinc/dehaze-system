package job

import "time"

// 定时任务分布式锁键与 TTL（调度关注点，由 JobHandler 持有）
const (
	orderJobLockTTL            = 5 * time.Minute
	memberQuotaResetJobLockTTL = 10 * time.Minute

	jobOrderCancelExpiredLockKey   = "job:order:cancel_expired:lock"
	jobOrderCompleteExpiredLockKey = "job:order:complete_expired:lock"
	jobOrderAutoRenewLockKey       = "job:order:auto_renew:lock"
	jobOrderExpireCouponsLockKey   = "job:order:expire_coupons:lock"
	jobOrderRefundRetryLockKey     = "job:order:refund_retry:lock"
	jobMemberQuotaResetLockKey     = "job:member:quota:reset:lock"
)

package job

import (
	"context"
	"time"

	"github.com/earthyzinc/clarity-go/pkg/cache/types"
	"github.com/earthyzinc/clarity-go/pkg/logger"
	"go.uber.org/zap"
)

// 定时任务分布式锁键与 TTL（调度关注点，由 JobHandler 持有）
const (
	orderJobLockTTL   = 5 * time.Minute
	vipGiftJobLockTTL = 10 * time.Minute

	jobOrderCancelExpiredLockKey   = "job:order:cancel_expired:lock"
	jobOrderCompleteExpiredLockKey = "job:order:complete_expired:lock"
	jobOrderAutoRenewLockKey       = "job:order:auto_renew:lock"
	jobOrderExpireCouponsLockKey   = "job:order:expire_coupons:lock"
	jobOrderRefundRetryLockKey     = "job:order:refund_retry:lock"
	jobVipGiftGrantLockKey         = "job:vip:gift:grant:lock"
	jobVipGiftClearLockKey         = "job:vip:gift:clear:lock"
)

// tryLock 尝试获取分布式锁，返回释放函数；cache 为 nil 时视为获取成功（本地/测试环境无 Redis）。
func tryLock(ctx context.Context, cache types.ICache, key string, ttl time.Duration) (func(), bool) {
	if cache == nil {
		return func() {}, true
	}
	token, ok, _ := cache.Lock(ctx, key, ttl)
	if !ok {
		logger.Debug("定时任务已被其他实例持有，跳过执行", zap.String("lock", key))
		return nil, false
	}
	return func() { _, _ = cache.Unlock(ctx, key, token) }, true
}

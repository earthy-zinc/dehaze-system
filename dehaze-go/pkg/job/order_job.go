package job

import (
	"context"
	"time"

	xxl "github.com/xxl-job/xxl-job-executor-go"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

// OrderJobRunner 订单定时任务执行接口（纯业务逻辑，无调度锁）
type OrderJobRunner interface {
	CancelExpiredOrders(ctx context.Context) error
	CompleteExpiredOrders(ctx context.Context) error
	ProcessAutoRenewals(ctx context.Context) error
	ExpireUserCoupons(ctx context.Context) error
	RetryFailedRefunds(ctx context.Context) error
}

// OrderJob 订单模块定时任务 — 负责分布式锁获取与编排，业务逻辑在 Service
type OrderJob struct {
	runner OrderJobRunner
	cache  types.ICache
}

// NewOrderJob 创建订单定时任务
func NewOrderJob(runner OrderJobRunner, cache types.ICache) *OrderJob {
	return &OrderJob{runner: runner, cache: cache}
}

// tryLock 尝试获取分布式锁，返回释放函数；获取失败时返回 nil + false
func (j *OrderJob) tryLock(ctx context.Context, key string, ttl time.Duration) (func(), bool) {
	if j.cache == nil {
		return func() {}, true
	}
	token, ok, _ := j.cache.Lock(ctx, key, ttl)
	if !ok {
		logger.Debug("定时任务已被其他实例持有，跳过执行", zap.String("lock", key))
		return nil, false
	}
	return func() { _, _ = j.cache.Unlock(ctx, key, token) }, true
}

func (j *OrderJob) HandleExpireOrders(ctx context.Context, param *xxl.RunReq) string {
	release, ok := j.tryLock(ctx, jobOrderCancelExpiredLockKey, orderJobLockTTL)
	if !ok {
		return "skipped: locked"
	}
	defer release()
	if err := j.runner.CancelExpiredOrders(ctx); err != nil {
		logger.Error("取消超时订单失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

func (j *OrderJob) HandleCompleteExpiredOrders(ctx context.Context, param *xxl.RunReq) string {
	release, ok := j.tryLock(ctx, jobOrderCompleteExpiredLockKey, orderJobLockTTL)
	if !ok {
		return "skipped: locked"
	}
	defer release()
	if err := j.runner.CompleteExpiredOrders(ctx); err != nil {
		logger.Error("归档到期订单失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

func (j *OrderJob) HandleAutoRenew(ctx context.Context, param *xxl.RunReq) string {
	release, ok := j.tryLock(ctx, jobOrderAutoRenewLockKey, orderJobLockTTL)
	if !ok {
		return "skipped: locked"
	}
	defer release()
	if err := j.runner.ProcessAutoRenewals(ctx); err != nil {
		logger.Error("处理自动续费失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

func (j *OrderJob) HandleExpireUserCoupons(ctx context.Context, param *xxl.RunReq) string {
	release, ok := j.tryLock(ctx, jobOrderExpireCouponsLockKey, orderJobLockTTL)
	if !ok {
		return "skipped: locked"
	}
	defer release()
	if err := j.runner.ExpireUserCoupons(ctx); err != nil {
		logger.Error("过期优惠券标记失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

func (j *OrderJob) HandleRetryFailedRefunds(ctx context.Context, param *xxl.RunReq) string {
	release, ok := j.tryLock(ctx, jobOrderRefundRetryLockKey, orderJobLockTTL)
	if !ok {
		return "skipped: locked"
	}
	defer release()
	if err := j.runner.RetryFailedRefunds(ctx); err != nil {
		logger.Error("退款失败重试失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

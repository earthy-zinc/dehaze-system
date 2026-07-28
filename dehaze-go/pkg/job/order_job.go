package job

import (
	"context"

	xxl "github.com/xxl-job/xxl-job-executor-go"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

// OrderJobRunner 订单定时任务执行接口
type OrderJobRunner interface {
	CancelExpiredOrders(ctx context.Context) error
	CompleteExpiredOrders(ctx context.Context) error
	ProcessAutoRenewals(ctx context.Context) error
	ExpireUserCoupons(ctx context.Context) error
}

// OrderJob 订单模块定时任务
type OrderJob struct {
	runner OrderJobRunner
}

// NewOrderJob 创建订单定时任务
func NewOrderJob(runner OrderJobRunner) *OrderJob {
	return &OrderJob{runner: runner}
}

func (j *OrderJob) HandleExpireOrders(ctx context.Context, param *xxl.RunReq) string {
	if err := j.runner.CancelExpiredOrders(ctx); err != nil {
		logger.Error("取消超时订单失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

func (j *OrderJob) HandleCompleteExpiredOrders(ctx context.Context, param *xxl.RunReq) string {
	if err := j.runner.CompleteExpiredOrders(ctx); err != nil {
		logger.Error("归档到期订单失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

func (j *OrderJob) HandleAutoRenew(ctx context.Context, param *xxl.RunReq) string {
	if err := j.runner.ProcessAutoRenewals(ctx); err != nil {
		logger.Error("处理自动续费失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

func (j *OrderJob) HandleExpireUserCoupons(ctx context.Context, param *xxl.RunReq) string {
	if err := j.runner.ExpireUserCoupons(ctx); err != nil {
		logger.Error("过期优惠券标记失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

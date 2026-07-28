package job

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/database"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

const (
	orderCancelInterval  = 1 * time.Minute
	autoRenewInterval    = 5 * time.Minute
	couponExpireInterval = 5 * time.Minute
)

// OrderJobRunner 订单定时任务执行接口
type OrderJobRunner interface {
	CancelExpiredOrders(ctx context.Context) error
	ProcessAutoRenewals(ctx context.Context) error
	ExpireUserCoupons(ctx context.Context) error
}

// OrderJob 订单模块定时任务
type OrderJob struct {
	running     bool
	runner      OrderJobRunner
	cancelFunc  context.CancelFunc
}

// NewOrderJob 创建订单定时任务
func NewOrderJob(runner OrderJobRunner) *OrderJob {
	return &OrderJob{runner: runner}
}

// Start 启动订单定时任务
func (j *OrderJob) Start() {
	if j.running {
		logger.Warn("订单定时任务已在运行")
		return
	}

	ctx, cancel := context.WithCancel(database.SetUserID(context.Background(), common.SystemUserID))
	j.cancelFunc = cancel
	j.running = true

	logger.Info("启动订单定时任务")
	go j.run(ctx)
}

// Stop 停止订单定时任务
func (j *OrderJob) Stop() {
	if !j.running {
		return
	}
	logger.Info("停止订单定时任务")
	j.cancelFunc()
	j.running = false
}

func (j *OrderJob) run(ctx context.Context) {
	cancelTicker := time.NewTicker(orderCancelInterval)
	defer cancelTicker.Stop()
	renewTicker := time.NewTicker(autoRenewInterval)
	defer renewTicker.Stop()
	couponTicker := time.NewTicker(couponExpireInterval)
	defer couponTicker.Stop()

	j.executeAll(ctx)

	for {
		select {
		case <-ctx.Done():
			logger.Info("订单定时任务已停止")
			return
		case <-cancelTicker.C:
			j.runCancelExpired(ctx)
		case <-renewTicker.C:
			j.runAutoRenew(ctx)
		case <-couponTicker.C:
			j.runExpireCoupons(ctx)
		}
	}
}

func (j *OrderJob) executeAll(ctx context.Context) {
	j.runCancelExpired(ctx)
	j.runAutoRenew(ctx)
	j.runExpireCoupons(ctx)
}

func (j *OrderJob) runCancelExpired(ctx context.Context) {
	if err := j.runner.CancelExpiredOrders(ctx); err != nil {
		logger.Error("取消超时订单失败", zap.Error(err))
	}
}

func (j *OrderJob) runAutoRenew(ctx context.Context) {
	if err := j.runner.ProcessAutoRenewals(ctx); err != nil {
		logger.Error("处理自动续费失败", zap.Error(err))
	}
}

func (j *OrderJob) runExpireCoupons(ctx context.Context) {
	if err := j.runner.ExpireUserCoupons(ctx); err != nil {
		logger.Error("过期优惠券标记失败", zap.Error(err))
	}
}

package job

import (
	"context"

	"github.com/earthyzinc/clarity-go/pkg/cache/types"
	"github.com/earthyzinc/clarity-go/pkg/logger"
	xxl "github.com/xxl-job/xxl-job-executor-go"
	"go.uber.org/zap"
)

// VipGiftRunner 会员月度积分赠送任务执行接口（纯业务逻辑，无调度锁）
type VipGiftRunner interface {
	GrantVipMonthlyGift(ctx context.Context) error
	ClearVipGiftExpire(ctx context.Context) error
}

// VipGiftJob VIP 月度积分赠送任务 — 负责分布式锁获取与编排，业务逻辑在 Service
type VipGiftJob struct {
	runner VipGiftRunner
	cache  types.ICache
}

func NewVipGiftJob(runner VipGiftRunner, cache types.ICache) *VipGiftJob {
	return &VipGiftJob{runner: runner, cache: cache}
}

func (j *VipGiftJob) HandleGrantVipMonthlyGift(ctx context.Context, param *xxl.RunReq) string {
	release, ok := tryLock(ctx, j.cache, jobVipGiftGrantLockKey, vipGiftJobLockTTL)
	if !ok {
		return "skipped: locked"
	}
	defer release()
	if err := j.runner.GrantVipMonthlyGift(ctx); err != nil {
		logger.Error("VIP月度赠送失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

func (j *VipGiftJob) HandleClearVipGiftExpire(ctx context.Context, param *xxl.RunReq) string {
	release, ok := tryLock(ctx, j.cache, jobVipGiftClearLockKey, vipGiftJobLockTTL)
	if !ok {
		return "skipped: locked"
	}
	defer release()
	if err := j.runner.ClearVipGiftExpire(ctx); err != nil {
		logger.Error("VIP赠送积分月末清零失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

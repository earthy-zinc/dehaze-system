package job

import (
	"context"
	"time"

	xxl "github.com/xxl-job/xxl-job-executor-go"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

// MemberJobRunner 会员模块定时任务执行接口（纯业务逻辑，无调度锁）
type MemberJobRunner interface {
	ResetMonthlyQuota(ctx context.Context) error
	ProcessExpiredMembers(ctx context.Context) error
	SendExpireReminders(ctx context.Context) error
}

// MemberJob 会员模块定时任务 — 负责分布式锁获取与编排，业务逻辑在 Service
type MemberJob struct {
	runner MemberJobRunner
	cache  types.ICache
}

// NewMemberJob 创建会员定时任务
func NewMemberJob(runner MemberJobRunner, cache types.ICache) *MemberJob {
	return &MemberJob{runner: runner, cache: cache}
}

// tryLock 尝试获取分布式锁，返回释放函数；获取失败时返回 nil + false
func (j *MemberJob) tryLock(ctx context.Context, key string, ttl time.Duration) (func(), bool) {
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

func (j *MemberJob) HandleResetMonthlyQuota(ctx context.Context, param *xxl.RunReq) string {
	release, ok := j.tryLock(ctx, jobMemberQuotaResetLockKey, memberQuotaResetJobLockTTL)
	if !ok {
		return "skipped: locked"
	}
	defer release()
	if err := j.runner.ResetMonthlyQuota(ctx); err != nil {
		logger.Error("重置会员月度配额失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

func (j *MemberJob) HandleProcessExpiredMembers(ctx context.Context, param *xxl.RunReq) string {
	if err := j.runner.ProcessExpiredMembers(ctx); err != nil {
		logger.Error("会员过期降级处理失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

func (j *MemberJob) HandleSendExpireReminders(ctx context.Context, param *xxl.RunReq) string {
	if err := j.runner.SendExpireReminders(ctx); err != nil {
		logger.Error("会员到期预警处理失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

package job

import (
	"context"

	xxl "github.com/xxl-job/xxl-job-executor-go"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

// MemberJobRunner 会员模块定时任务执行接口
type MemberJobRunner interface {
	ResetMonthlyQuota(ctx context.Context) error
	ProcessExpiredMembers(ctx context.Context) error
	SendExpireReminders(ctx context.Context) error
}

// MemberJob 会员模块定时任务
type MemberJob struct {
	runner MemberJobRunner
}

// NewMemberJob 创建会员定时任务
func NewMemberJob(runner MemberJobRunner) *MemberJob {
	return &MemberJob{runner: runner}
}

func (j *MemberJob) HandleResetMonthlyQuota(ctx context.Context, param *xxl.RunReq) string {
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

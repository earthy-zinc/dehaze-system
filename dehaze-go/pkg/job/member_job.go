package job

import (
	"context"

	xxl "github.com/xxl-job/xxl-job-executor-go"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

// MemberQuotaRunner 会员配额定时任务执行接口
type MemberQuotaRunner interface {
	ResetMonthlyQuota(ctx context.Context) error
}

// MemberJob 会员模块定时任务
type MemberJob struct {
	runner MemberQuotaRunner
}

// NewMemberJob 创建会员定时任务
func NewMemberJob(runner MemberQuotaRunner) *MemberJob {
	return &MemberJob{runner: runner}
}

func (j *MemberJob) HandleResetMonthlyQuota(ctx context.Context, param *xxl.RunReq) string {
	if err := j.runner.ResetMonthlyQuota(ctx); err != nil {
		logger.Error("重置会员月度配额失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

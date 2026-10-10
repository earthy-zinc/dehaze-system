package job

import (
	"context"

	"github.com/earthyzinc/dehaze-go/pkg/logger"
	xxl "github.com/xxl-job/xxl-job-executor-go"
	"go.uber.org/zap"
)

// MemberJobRunner 会员模块定时任务执行接口（纯业务逻辑，无调度锁）
type MemberJobRunner interface {
	ProcessExpiredMembers(ctx context.Context) error
	SendExpireReminders(ctx context.Context) error
}

// MemberJob 会员模块定时任务 — 负责编排，业务逻辑在 Service
type MemberJob struct {
	runner MemberJobRunner
}

// NewMemberJob 创建会员定时任务
func NewMemberJob(runner MemberJobRunner) *MemberJob {
	return &MemberJob{runner: runner}
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

package job

import (
	"context"

	xxl "github.com/xxl-job/xxl-job-executor-go"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

// AnnouncementRunner 公告定时任务执行接口
type AnnouncementRunner interface {
	SendScheduled(ctx context.Context) error
}

// AnnouncementJob 公告定时任务
type AnnouncementJob struct {
	runner AnnouncementRunner
}

// NewAnnouncementJob 创建公告定时任务
func NewAnnouncementJob(runner AnnouncementRunner) *AnnouncementJob {
	return &AnnouncementJob{runner: runner}
}

func (j *AnnouncementJob) HandleSendScheduledAnnouncements(ctx context.Context, param *xxl.RunReq) string {
	if err := j.runner.SendScheduled(ctx); err != nil {
		logger.Error("定时公告发送失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

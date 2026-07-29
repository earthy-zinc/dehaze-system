package job

import (
	"context"

	"github.com/earthyzinc/dehaze-go/pkg/logger"
	xxl "github.com/xxl-job/xxl-job-executor-go"
	"go.uber.org/zap"
)

// MessageRunner 消息定时任务执行接口
type MessageRunner interface {
	CleanupExpired(ctx context.Context) error
	RefreshUnreadCountCache(ctx context.Context) error
}

// MessageJob 消息定时任务
type MessageJob struct {
	runner MessageRunner
}

// NewMessageJob 创建消息定时任务
func NewMessageJob(runner MessageRunner) *MessageJob {
	return &MessageJob{runner: runner}
}

func (j *MessageJob) HandleCleanupExpiredMessages(ctx context.Context, param *xxl.RunReq) string {
	if err := j.runner.CleanupExpired(ctx); err != nil {
		logger.Error("清理过期消息失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

func (j *MessageJob) HandleRefreshUnreadCountCache(ctx context.Context, param *xxl.RunReq) string {
	if err := j.runner.RefreshUnreadCountCache(ctx); err != nil {
		logger.Error("未读数缓存刷新失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

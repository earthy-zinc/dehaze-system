package job

import (
	"context"

	xxl "github.com/xxl-job/xxl-job-executor-go"
	evalrepo "github.com/earthyzinc/dehaze-go/internal/repository/eval_log"
	predrepo "github.com/earthyzinc/dehaze-go/internal/repository/pred_log"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/database"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
)

func InitJobs(executor xxl.Executor, storageSvc StorageService, predLogRepo predrepo.IPredLogRepository, evalLogRepo evalrepo.IEvalLogRepository, orderRunner OrderJobRunner, announcementSvc AnnouncementRunner, messageSvc MessageRunner, memberRunner MemberQuotaRunner) {
	if executor == nil {
		logger.Warn("XXL-Job 执行器未初始化，跳过任务注册")
		return
	}

	cleanupJob := NewCleanupJob(storageSvc, predLogRepo, evalLogRepo)
	orderJob := NewOrderJob(orderRunner)
	announcementJob := NewAnnouncementJob(announcementSvc)
	messageJob := NewMessageJob(messageSvc)
	memberJob := NewMemberJob(memberRunner)

	executor.RegTask("cleanupExpiredTasks", cleanupJob.HandleCleanupExpiredTasks)
	executor.RegTask("cleanupStuckTasks", cleanupJob.HandleCleanupStuckTasks)
	executor.RegTask("cleanupStuckPredEvalLogs", cleanupJob.HandleCleanupStuckPredEvalLogs)
	executor.RegTask("expireOrders", orderJob.HandleExpireOrders)
	executor.RegTask("completeExpiredOrders", orderJob.HandleCompleteExpiredOrders)
	executor.RegTask("autoRenew", orderJob.HandleAutoRenew)
	executor.RegTask("expireUserCoupons", orderJob.HandleExpireUserCoupons)
	executor.RegTask("sendScheduledAnnouncements", announcementJob.HandleSendScheduledAnnouncements)
	executor.RegTask("cleanupExpiredMessages", messageJob.HandleCleanupExpiredMessages)
	executor.RegTask("resetMemberMonthlyQuota", memberJob.HandleResetMonthlyQuota)

	logger.Info("XXL-Job 定时任务注册完成")
}

func SystemContext() context.Context {
	return database.SetUserID(context.Background(), common.SystemUserID)
}

func StopJobs() {
	logger.Info("定时任务已停止")
}

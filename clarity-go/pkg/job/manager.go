package job

import (
	"context"

	evalrepo "github.com/earthyzinc/clarity-go/internal/repository/eval_log"
	predrepo "github.com/earthyzinc/clarity-go/internal/repository/pred_log"
	"github.com/earthyzinc/clarity-go/pkg/cache/types"
	"github.com/earthyzinc/clarity-go/pkg/common"
	"github.com/earthyzinc/clarity-go/pkg/database"
	"github.com/earthyzinc/clarity-go/pkg/logger"
	"github.com/earthyzinc/clarity-go/pkg/trace"
	xxl "github.com/xxl-job/xxl-job-executor-go"
	"go.uber.org/zap"
)

// wrapWithTrace 为后台任务注入 job 级 trace_id，使日志可追踪
func wrapWithTrace(handler func(ctx context.Context, param *xxl.RunReq) string) func(ctx context.Context, param *xxl.RunReq) string {
	return func(ctx context.Context, param *xxl.RunReq) string {
		traceID := trace.NewTraceID()
		ctx = trace.WithTraceID(ctx, traceID)
		ctx = trace.WithLogger(ctx, zap.L().With(zap.String(trace.TraceFieldName, traceID)))
		return handler(ctx, param)
	}
}

func InitJobs(executor xxl.Executor, storageSvc StorageService, predLogRepo predrepo.IPredLogRepository, evalLogRepo evalrepo.IEvalLogRepository, orderRunner OrderJobRunner, announcementSvc AnnouncementRunner, messageSvc MessageRunner, memberRunner MemberJobRunner, vipGiftRunner VipGiftRunner, cache types.ICache) {
	if executor == nil {
		logger.Warn("XXL-Job 执行器未初始化，跳过任务注册")
		return
	}

	cleanupJob := NewCleanupJob(storageSvc, predLogRepo, evalLogRepo)
	orderJob := NewOrderJob(orderRunner, cache)
	announcementJob := NewAnnouncementJob(announcementSvc)
	messageJob := NewMessageJob(messageSvc)
	memberJob := NewMemberJob(memberRunner)
	vipGiftJob := NewVipGiftJob(vipGiftRunner, cache)

	executor.RegTask("cleanupExpiredTasks", wrapWithTrace(cleanupJob.HandleCleanupExpiredTasks))
	executor.RegTask("cleanupStuckTasks", wrapWithTrace(cleanupJob.HandleCleanupStuckTasks))
	executor.RegTask("cleanupStuckPredEvalLogs", wrapWithTrace(cleanupJob.HandleCleanupStuckPredEvalLogs))
	executor.RegTask("expireOrders", wrapWithTrace(orderJob.HandleExpireOrders))
	executor.RegTask("completeExpiredOrders", wrapWithTrace(orderJob.HandleCompleteExpiredOrders))
	executor.RegTask("autoRenew", wrapWithTrace(orderJob.HandleAutoRenew))
	executor.RegTask("expireUserCoupons", wrapWithTrace(orderJob.HandleExpireUserCoupons))
	executor.RegTask("sendScheduledAnnouncements", wrapWithTrace(announcementJob.HandleSendScheduledAnnouncements))
	executor.RegTask("cleanupExpiredMessages", wrapWithTrace(messageJob.HandleCleanupExpiredMessages))
	executor.RegTask("refreshUnreadCountCache", wrapWithTrace(messageJob.HandleRefreshUnreadCountCache))
	executor.RegTask("grantVipMonthlyGift", wrapWithTrace(vipGiftJob.HandleGrantVipMonthlyGift))
	executor.RegTask("clearVipGiftExpire", wrapWithTrace(vipGiftJob.HandleClearVipGiftExpire))
	executor.RegTask("processExpiredMembers", wrapWithTrace(memberJob.HandleProcessExpiredMembers))
	executor.RegTask("retryFailedRefunds", wrapWithTrace(orderJob.HandleRetryFailedRefunds))
	executor.RegTask("sendExpireReminders", wrapWithTrace(memberJob.HandleSendExpireReminders))

	logger.Info("XXL-Job 定时任务注册完成")
}

func SystemContext() context.Context {
	ctx := trace.WithTraceID(context.Background(), trace.NewTraceID())
	ctx = trace.WithLogger(ctx, zap.L().With(zap.String(trace.TraceFieldName, trace.FromContext(ctx))))
	return database.SetUserID(ctx, common.SystemUserID)
}

func StopJobs() {
	logger.Info("定时任务已停止")
}

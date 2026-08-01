package job

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	evalrepo "github.com/earthyzinc/dehaze-go/internal/repository/eval_log"
	predrepo "github.com/earthyzinc/dehaze-go/internal/repository/pred_log"
	tasksvc "github.com/earthyzinc/dehaze-go/internal/service/task"
	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/database"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	xxl "github.com/xxl-job/xxl-job-executor-go"
	"go.uber.org/zap"
)

const (
	// 失败记录保留时间（7天）
	FailedRecordRetention = 7 * 24 * time.Hour

	// 任务清理阈值（与 Python 端对齐）
	expiredThreshold = 7 * 24 * time.Hour  // 7天前已完成/取消任务清理
	oldThreshold     = 30 * 24 * time.Hour // 30天前已终止任务清理
	stuckProcessing  = 30 * time.Minute    // PROCESSING 30分钟无进展
	stuckPending     = 24 * time.Hour      // PENDING 24小时未启动

	// 预测/评估僵尸任务清理（对齐 Java PredEvalLogCleanupJob）
	predEvalStuckThreshold = 10 * time.Minute
)

// CleanupJob 清理任务管理器
type CleanupJob struct {
	cacheClient    types.ICache
	storageService StorageService
	predLogRepo    predrepo.IPredLogRepository
	evalLogRepo    evalrepo.IEvalLogRepository
}

// StorageService 存储服务接口（仅声明 cleanup 所需方法）
type StorageService interface {
	Exists(ctx context.Context, objectName string) (bool, error)
}

// NewCleanupJob 创建清理任务管理器
func NewCleanupJob(storageSvc StorageService, predLogRepo predrepo.IPredLogRepository, evalLogRepo evalrepo.IEvalLogRepository) *CleanupJob {
	return &CleanupJob{
		cacheClient:    cache.GetCache(),
		storageService: storageSvc,
		predLogRepo:    predLogRepo,
		evalLogRepo:    evalLogRepo,
	}
}

func (j *CleanupJob) HandleCleanupExpiredTasks(ctx context.Context, param *xxl.RunReq) string {
	j.executeCleanup(ctx)
	return "success"
}

func (j *CleanupJob) HandleCleanupStuckTasks(ctx context.Context, param *xxl.RunReq) string {
	if err := j.cleanupStuckTasks(ctx); err != nil {
		logger.Error("清理僵死任务失败", zap.Error(err))
		return "failed: " + err.Error()
	}
	return "success"
}

func (j *CleanupJob) HandleCleanupStuckPredEvalLogs(ctx context.Context, param *xxl.RunReq) string {
	j.cleanupStuckPredEvalLogs(ctx)
	return "success"
}

// cleanupStuckPredEvalLogs 回收预测/评估僵尸任务
// 扫描 status='processing' AND update_time < NOW() - 10min 的记录，标记为 failed
func (j *CleanupJob) cleanupStuckPredEvalLogs(ctx context.Context) {
	threshold := time.Now().Add(-predEvalStuckThreshold)

	predCount, err := j.predLogRepo.MarkStuckAsFailed(ctx, threshold)
	if err != nil {
		logger.Error("回收预测僵尸任务失败", zap.Error(err))
	} else if predCount > 0 {
		logger.Warn("回收预测僵尸任务", zap.Int("count", predCount))
	}

	evalCount, err := j.evalLogRepo.MarkStuckAsFailed(ctx, threshold)
	if err != nil {
		logger.Error("回收评估僵尸任务失败", zap.Error(err))
	} else if evalCount > 0 {
		logger.Warn("回收评估僵尸任务", zap.Int("count", evalCount))
	}
}

// executeCleanup 执行清理操作
// 各步骤独立执行，某一步失败不阻断后续步骤
func (j *CleanupJob) executeCleanup(ctx context.Context) {
	logger.Debug("开始执行清理任务")

	// 1. 清理失败的缩略图生成记录
	if err := j.cleanupFailedThumbnails(ctx); err != nil {
		logger.Error("清理缩略图失败记录失败", zap.Error(err))
	}

	// 2. 清理失败的文件删除记录
	if err := j.cleanupFailedDeletions(ctx); err != nil {
		logger.Error("清理删除失败记录失败", zap.Error(err))
	}

	// 3. 清理过期任务（对齐 Python cleanupExpiredTasks）
	if err := j.cleanupExpiredTasks(ctx); err != nil {
		logger.Error("清理过期任务失败", zap.Error(err))
	}

	logger.Debug("清理任务执行完成")
}

// RunCleanupNow 立即执行一次清理（用于手动触发）
func (j *CleanupJob) RunCleanupNow() error {
	ctx, cancel := context.WithTimeout(database.SetUserID(context.Background(), common.SystemUserID), 5*time.Minute)
	defer cancel()

	j.executeCleanup(ctx)
	return nil
}

// cleanupFailedThumbnails 清理失败的缩略图生成记录
func (j *CleanupJob) cleanupFailedThumbnails(ctx context.Context) error {
	const THUMBNAIL_FAILED_KEY = "dataset:thumbnail:failed"

	entries, err := j.cacheClient.HGetAll(ctx, THUMBNAIL_FAILED_KEY)
	if err != nil {
		return fmt.Errorf("获取失败记录: %w", err)
	}

	if len(entries) == 0 {
		return nil
	}

	threshold := time.Now().Add(-FailedRecordRetention).Unix()
	cleanedCount := 0

	for fileIDStr, timestampStr := range entries {
		var fileID int64
		var timestamp int64

		if _, err := fmt.Sscanf(fileIDStr, "%d", &fileID); err != nil {
			continue
		}
		if _, err := fmt.Sscanf(timestampStr, "%d", &timestamp); err != nil {
			continue
		}

		if timestamp < threshold {
			j.cacheClient.HDel(ctx, THUMBNAIL_FAILED_KEY, fileIDStr)
			cleanedCount++
			logger.Debug("清理过期的缩略图失败记录",
				zap.Int64("fileID", fileID),
				zap.Int64("ageDays", (time.Now().Unix()-timestamp)/86400))
		}
	}

	if cleanedCount > 0 {
		logger.Debug("清理缩略图失败记录完成", zap.Int("count", cleanedCount))
	}

	return nil
}

// cleanupFailedDeletions 清理失败的文件删除记录
//
// 检查文件是否仍存在于存储中：
// - 文件已不存在 → 删除成功，从失败列表移除
// - 文件仍存在 → 保留记录，等待后续重试
func (j *CleanupJob) cleanupFailedDeletions(ctx context.Context) error {
	const DELETION_FAILED_KEY = "dataset:deletion:failed"

	members, err := j.cacheClient.SMembers(ctx, DELETION_FAILED_KEY)
	if err != nil {
		return fmt.Errorf("获取删除失败记录: %w", err)
	}

	if len(members) == 0 {
		return nil
	}

	storageService := j.storageService
	cleanedCount := 0
	for _, filePath := range members {
		if len(filePath) == 0 {
			j.cacheClient.SRem(ctx, DELETION_FAILED_KEY, filePath)
			continue
		}

		// 检查文件是否仍存在
		exists, existsErr := storageService.Exists(ctx, filePath)
		if existsErr != nil {
			logger.Warn("检查文件存在性失败，保留记录",
				zap.String("path", filePath), zap.Error(existsErr))
			continue
		}
		if !exists {
			// 文件已不存在，删除成功，从失败列表移除
			j.cacheClient.SRem(ctx, DELETION_FAILED_KEY, filePath)
			cleanedCount++
			logger.Debug("从删除失败列表移除（文件已不存在）", zap.String("path", filePath))
		}
	}

	if cleanedCount > 0 {
		logger.Debug("清理删除失败记录完成", zap.Int("count", cleanedCount))
	}

	return nil
}

// cleanupExpiredTasks 清理过期任务（对齐 Python cleanupExpiredTasks）
// 1. 7天前 COMPLETED/CANCELLED 任务物理删除
// 2. 30天前所有非 PENDING/PROCESSING 任务物理删除
func (j *CleanupJob) cleanupExpiredTasks(ctx context.Context) error {
	db := database.DB()
	if db == nil {
		return fmt.Errorf("数据库未初始化")
	}

	// Block 1: 7天前 COMPLETED/CANCELLED 任务
	sevenDaysAgo := time.Now().Add(-expiredThreshold)
	var block1Tasks []model.SysTask
	if err := db.WithContext(ctx).
		Where("status IN ?", []model.TaskStatus{model.TaskStatusCompleted, model.TaskStatusCancelled}).
		Where("create_time < ?", sevenDaysAgo).
		Find(&block1Tasks).Error; err != nil {
		return fmt.Errorf("查询7天前已完成/取消任务失败: %w", err)
	}

	if len(block1Tasks) > 0 {
		ids := make([]int64, len(block1Tasks))
		cacheKeys := make([]string, 0, len(block1Tasks))
		for i, t := range block1Tasks {
			ids[i] = t.ID
			if t.TaskID != "" {
				cacheKeys = append(cacheKeys, tasksvc.TASK_CACHE_PREFIX+t.TaskID)
			}
		}

		// 删除 DB 记录
		if err := db.WithContext(ctx).Where("id IN ?", ids).Delete(&model.SysTask{}).Error; err != nil {
			return fmt.Errorf("删除7天前已完成/取消任务失败: %w", err)
		}

		// 删除 Redis 缓存
		if len(cacheKeys) > 0 {
			j.cacheClient.Delete(ctx, cacheKeys...)
		}

		logger.Debug("清理7天前已完成/取消任务", zap.Int("count", len(block1Tasks)))
	}

	// Block 2: 30天前所有非 PENDING/PROCESSING 任务
	thirtyDaysAgo := time.Now().Add(-oldThreshold)
	var block2Tasks []model.SysTask
	if err := db.WithContext(ctx).
		Where("status NOT IN ?", []model.TaskStatus{model.TaskStatusPending, model.TaskStatusProcessing}).
		Where("create_time < ?", thirtyDaysAgo).
		Find(&block2Tasks).Error; err != nil {
		return fmt.Errorf("查询30天前已终止任务失败: %w", err)
	}

	if len(block2Tasks) > 0 {
		ids := make([]int64, len(block2Tasks))
		cacheKeys := make([]string, 0, len(block2Tasks))
		for i, t := range block2Tasks {
			ids[i] = t.ID
			if t.TaskID != "" {
				cacheKeys = append(cacheKeys, tasksvc.TASK_CACHE_PREFIX+t.TaskID)
			}
		}

		if err := db.WithContext(ctx).Where("id IN ?", ids).Delete(&model.SysTask{}).Error; err != nil {
			return fmt.Errorf("删除30天前已终止任务失败: %w", err)
		}

		if len(cacheKeys) > 0 {
			j.cacheClient.Delete(ctx, cacheKeys...)
		}

		logger.Debug("清理30天前已终止任务", zap.Int("count", len(block2Tasks)))
	}

	return nil
}

// cleanupStuckTasks 回收僵死任务（对齐 Python cleanupStuckTasks）
// 1. PROCESSING 且 startedAt < 30分钟 → FAILED
// 2. PENDING 且 createTime < 24小时 → FAILED
func (j *CleanupJob) cleanupStuckTasks(ctx context.Context) error {
	db := database.DB()
	if db == nil {
		return fmt.Errorf("数据库未初始化")
	}

	now := time.Now()

	// Block 1: PROCESSING 且 startedAt < 30分钟
	thirtyMinAgo := now.Add(-stuckProcessing)
	var stuckProcessing []model.SysTask
	if err := db.WithContext(ctx).
		Where("status = ?", model.TaskStatusProcessing).
		Where("started_at < ?", thirtyMinAgo).
		Find(&stuckProcessing).Error; err != nil {
		return fmt.Errorf("查询PROCESSING僵死任务失败: %w", err)
	}

	if len(stuckProcessing) > 0 {
		errorMsg := "任务超时（30分钟无进度更新），已被系统自动回收"
		for _, t := range stuckProcessing {
			if err := db.WithContext(ctx).Model(&model.SysTask{}).
				Where("id = ?", t.ID).
				Updates(map[string]interface{}{
					"status":        model.TaskStatusFailed,
					"error_message": errorMsg,
					"completed_at":  now,
				}).Error; err != nil {
				logger.Error("更新PROCESSING僵死任务失败",
					zap.Int64("taskID", t.ID), zap.Error(err))
				continue
			}

			// 更新 Redis 缓存
			if t.TaskID != "" {
				t.Status = model.TaskStatusFailed
				t.ErrorMessage = errorMsg
				t.CompletedAt = &now
				j.cacheTask(ctx, &t)
			}
		}
		logger.Warn("清理PROCESSING僵死任务", zap.Int("count", len(stuckProcessing)))
	}

	// Block 2: PENDING 且 createTime < 24小时
	oneDayAgo := now.Add(-stuckPending)
	var stuckPending []model.SysTask
	if err := db.WithContext(ctx).
		Where("status = ?", model.TaskStatusPending).
		Where("create_time < ?", oneDayAgo).
		Find(&stuckPending).Error; err != nil {
		return fmt.Errorf("查询PENDING僵死任务失败: %w", err)
	}

	if len(stuckPending) > 0 {
		errorMsg := "任务超时（24h未启动），已被系统自动回收"
		for _, t := range stuckPending {
			if err := db.WithContext(ctx).Model(&model.SysTask{}).
				Where("id = ?", t.ID).
				Updates(map[string]interface{}{
					"status":        model.TaskStatusFailed,
					"error_message": errorMsg,
					"completed_at":  now,
				}).Error; err != nil {
				logger.Error("更新PENDING僵死任务失败",
					zap.Int64("taskID", t.ID), zap.Error(err))
				continue
			}

			if t.TaskID != "" {
				t.Status = model.TaskStatusFailed
				t.ErrorMessage = errorMsg
				t.CompletedAt = &now
				j.cacheTask(ctx, &t)
			}
		}
		logger.Warn("清理PENDING僵死任务", zap.Int("count", len(stuckPending)))
	}

	return nil
}

// cacheTask 缓存任务到 Redis（须与 task_service.go 的 cacheTask 逻辑一致）
func (j *CleanupJob) cacheTask(ctx context.Context, task *model.SysTask) {
	cacheKey := tasksvc.TASK_CACHE_PREFIX + task.TaskID
	taskJSON, err := json.Marshal(task)
	if err != nil {
		logger.Error("任务序列化失败", zap.Error(err))
		return
	}
	if err := j.cacheClient.Set(ctx, cacheKey, string(taskJSON), tasksvc.TASK_EXPIRE_HOURS); err != nil {
		logger.Error("缓存任务失败", zap.Error(err))
	}
}

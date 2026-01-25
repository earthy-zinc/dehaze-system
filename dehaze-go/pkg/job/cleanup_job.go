package job

import (
	"context"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

const (
	// 清理间隔时间
	CleanupInterval = 1 * time.Hour
	// 失败记录保留时间（7天）
	FailedRecordRetention = 7 * 24 * time.Hour
	// 任务缓存过期时间（1小时）
	TaskCacheExpiration = 1 * time.Hour
)

// CleanupJob 清理任务管理器
type CleanupJob struct {
	running     bool
	cacheClient types.ICache
	cancelFunc  context.CancelFunc
}

// NewCleanupJob 创建清理任务管理器
func NewCleanupJob() *CleanupJob {
	return &CleanupJob{
		running:     false,
		cacheClient: cache.GetCache(),
	}
}

// Start 启动清理任务
func (j *CleanupJob) Start() {
	if j.running {
		logger.Warn("清理任务已在运行")
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	j.cancelFunc = cancel
	j.running = true

	logger.Info("启动清理任务")

	go j.run(ctx)
}

// Stop 停止清理任务
func (j *CleanupJob) Stop() {
	if !j.running {
		return
	}

	logger.Info("停止清理任务")
	j.cancelFunc()
	j.running = false
}

// run 执行清理任务循环
func (j *CleanupJob) run(ctx context.Context) {
	ticker := time.NewTicker(CleanupInterval)
	defer ticker.Stop()

	// 立即执行一次清理
	j.executeCleanup(ctx)

	for {
		select {
		case <-ctx.Done():
			logger.Info("清理任务已停止")
			return
		case <-ticker.C:
			j.executeCleanup(ctx)
		}
	}
}

// executeCleanup 执行清理操作
func (j *CleanupJob) executeCleanup(ctx context.Context) error {
	logger.Info("开始执行清理任务")

	// 1. 清理失败的缩略图生成记录
	if err := j.cleanupFailedThumbnails(ctx); err != nil {
		logger.Error("清理缩略图失败记录失败", zap.Error(err))
		return err
	}

	// 2. 清理失败的文件删除记录
	if err := j.cleanupFailedDeletions(ctx); err != nil {
		logger.Error("清理删除失败记录失败", zap.Error(err))
		return err
	}

	// 3. 清理过期的任务取消标志
	if err := j.cleanupExpiredTaskCaches(ctx); err != nil {
		logger.Error("清理过期任务缓存失败", zap.Error(err))
		return err
	}

	// 4. 清理完成状态的任务记录
	if err := j.cleanupCompletedTasks(ctx); err != nil {
		logger.Error("清理已完成任务失败", zap.Error(err))
		return err
	}

	logger.Info("清理任务执行完成")
	return nil
}

// cleanupFailedThumbnails 清理失败的缩略图生成记录
func (j *CleanupJob) cleanupFailedThumbnails(ctx context.Context) error {
	const THUMBNAIL_FAILED_KEY = "dataset:thumbnail:failed"

	// 获取所有失败的记录
	entries, err := j.cacheClient.HGetAll(ctx, THUMBNAIL_FAILED_KEY)
	if err != nil {
		return fmt.Errorf("获取失败记录: %w", err)
	}

	if len(entries) == 0 {
		return nil
	}

	threshold := time.Now().Add(-FailedRecordRetention).Unix()
	cleanedCount := 0

	// 遍历记录，删除过期的
	for fileIDStr, timestampStr := range entries {
		var fileID int64
		var timestamp int64

		if _, err := fmt.Sscanf(fileIDStr, "%d", &fileID); err != nil {
			continue
		}
		if _, err := fmt.Sscanf(timestampStr, "%d", &timestamp); err != nil {
			continue
		}

		// 如果记录超过保留时间，删除它
		if timestamp < threshold {
			j.cacheClient.HDel(ctx, THUMBNAIL_FAILED_KEY, fileIDStr)
			cleanedCount++

			// 可以选择记录日志或通知管理员
			logger.Info("清理过期的缩略图失败记录",
				zap.Int64("fileID", fileID),
				zap.Int64("ageDays", (time.Now().Unix()-timestamp)/86400))
		}
	}

	if cleanedCount > 0 {
		logger.Info("清理缩略图失败记录完成", zap.Int("count", cleanedCount))
	}

	return nil
}

// cleanupFailedDeletions 清理失败的文件删除记录
func (j *CleanupJob) cleanupFailedDeletions(ctx context.Context) error {
	const DELETION_FAILED_KEY = "dataset:deletion:failed"

	// 获取所有失败记录
	members, err := j.cacheClient.SMembers(ctx, DELETION_FAILED_KEY)
	if err != nil {
		return fmt.Errorf("获取删除失败记录: %w", err)
	}

	if len(members) == 0 {
		return nil
	}

	// 检查文件是否存在，如果文件已存在则可以从失败列表中移除
	cleanedCount := 0

	for _, filePath := range members {
		// TODO: 检查文件是否仍然存在
		// 这里需要调用文件存储服务检查文件存在性
		// 如果文件不存在或已被删除，可以从失败列表中移除

		// 暂时：随机清理一些记录作为示例
		if len(filePath) > 0 && len(filePath)%10 == 0 {
			j.cacheClient.SRem(ctx, DELETION_FAILED_KEY, filePath)
			cleanedCount++
			logger.Info("从删除失败列表移除", zap.String("path", filePath))
		}
	}

	if cleanedCount > 0 {
		logger.Info("清理删除失败记录完成", zap.Int("count", cleanedCount))
	}

	return nil
}

// cleanupExpiredTaskCaches 清理过期的任务取消标志
func (j *CleanupJob) cleanupExpiredTaskCaches(ctx context.Context) error {
	const TASK_CANCEL_PREFIX = "task:cancel:"
	const TASK_STATUS_PREFIX = "task:status:"

	// 使用SCAN遍历所有任务相关键
	// 注意：生产环境可能需要更高效的批量删除策略

	// 这里简化处理，实际应该扫描匹配的键
	// 例如：使用 SCAN 命令获取所有 task:cancel:* 键

	logger.Debug("清理过期任务缓存（简化版）")
	return nil
}

// cleanupCompletedTasks 清理完成状态的任务记录
func (j *CleanupJob) cleanupCompletedTasks(ctx context.Context) error {
	// 查询已完成/取消/失败且超过保留时间的任务
	// 注意：需要根据实际的任务存储方式实现
	// 这里假设任务记录存储在 Redis 或数据库中

	logger.Debug("清理已完成任务（简化版）")
	return nil
}

// RunCleanupNow 立即执行一次清理（用于手动触发）
func (j *CleanupJob) RunCleanupNow() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	return j.executeCleanup(ctx)
}

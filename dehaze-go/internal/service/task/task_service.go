package task

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	datasetrepo "github.com/earthyzinc/dehaze-go/internal/repository/dataset"
	taskrepo "github.com/earthyzinc/dehaze-go/internal/repository/task"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/database"
	"github.com/earthyzinc/dehaze-go/pkg/websocket"
	"github.com/google/uuid"
	"go.uber.org/zap"
)

const (
	TASK_CACHE_PREFIX     = "export:task:"
	TASK_CANCEL_PREFIX    = "task:cancel:"
	TASK_EXPIRE_HOURS     = 24 * time.Hour
	TASK_CANCEL_EXPIRE_MS = 5 * time.Minute

	// 幂等键 Redis 缓存前缀（映射 idempotency_key -> task_id）
	IDEMPOTENCY_KEY_PREFIX      = "idempotency:task:"
	IDEMPOTENCY_KEY_TTL         = 24 * time.Hour
)

// TaskService 任务服务
type TaskService struct {
	taskRepo     taskrepo.ITaskRepository
	datasetRepo  datasetrepo.IDatasetRepository
	cache        types.ICache
	logger       *zap.Logger
	taskExecutor AsyncTaskExecutor
}

// NewTaskService 创建任务服务
func NewTaskService(
	taskRepo taskrepo.ITaskRepository,
	datasetRepo datasetrepo.IDatasetRepository,
	c types.ICache,
	logger *zap.Logger,
	taskExecutor AsyncTaskExecutor,
) *TaskService {
	return &TaskService{
		taskRepo:     taskRepo,
		datasetRepo:  datasetRepo,
		cache:        c,
		logger:       logger,
		taskExecutor: taskExecutor,
	}
}

// GetTaskExecutor 获取任务执行器（用于关闭）
func (ts *TaskService) GetTaskExecutor() AsyncTaskExecutor {
	return ts.taskExecutor
}

// CreateExportTask 创建导出任务
// 统一任务接口为异步执行：同步创建任务记录（PENDING），资源存在性校验在异步策略中执行。
// 即使任务执行器发布失败，也不阻断任务创建，仅记录日志，任务最终由异步策略或超时清理处理。
// idempotencyKey 为客户端幂等键（可空），相同键返回已有任务，避免重复创建。
func (ts *TaskService) CreateExportTask(ctx context.Context, form bo.ExportTaskCreateForm, userID int64, idempotencyKey string) (*model.SysTask, error) {
	ctx = database.SetUserID(ctx, userID)

	// 幂等去重：若提供了 idempotencyKey，先查 DB 再查 Redis 缓存
	if idempotencyKey != "" {
		existing, err := ts.taskRepo.FindByIdempotencyKey(ctx, idempotencyKey)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询幂等任务失败", err)
		}
		if existing != nil {
			ts.logger.Info("幂等键命中，返回已有任务",
				zap.String("idempotencyKey", idempotencyKey),
				zap.String("taskID", existing.TaskID))
			return existing, nil
		}

		// 并发兜底：查 Redis 缓存
		cacheKey := IDEMPOTENCY_KEY_PREFIX + idempotencyKey
		if cachedTaskID, _ := ts.cache.Get(ctx, cacheKey); cachedTaskID != "" {
			if cachedTask, _ := ts.taskRepo.FindByTaskID(ctx, cachedTaskID); cachedTask != nil {
				return cachedTask, nil
			}
		}
	}

	taskIDStr := uuid.New().String()

	paramsJSON, err := json.Marshal(form)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "参数序列化失败", err)
	}

	now := time.Now()
	expiresAt := now.Add(TASK_EXPIRE_HOURS)

	task := model.SysTask{
		TaskID:         taskIDStr,
		TaskType:       model.TaskTypeDatasetExport,
		Status:         model.TaskStatusPending,
		Progress:       0,
		TotalFiles:     0,
		ProcessedFiles: 0,
		Params:         string(paramsJSON),
		ExpiresAt:      &expiresAt,
		IdempotencyKey: nil,
	}
	if idempotencyKey != "" {
		task.IdempotencyKey = &idempotencyKey
	}

	err = ts.taskRepo.Create(ctx, &task)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "保存任务失败", err)
	}

	ts.cacheTask(ctx, &task)

	// 缓存幂等键映射
	if idempotencyKey != "" {
		idempotencyCacheKey := IDEMPOTENCY_KEY_PREFIX + idempotencyKey
		if err := ts.cache.Set(ctx, idempotencyCacheKey, taskIDStr, IDEMPOTENCY_KEY_TTL); err != nil {
			ts.logger.Warn("缓存幂等键失败", zap.String("idempotencyKey", idempotencyKey), zap.Error(err))
		}
	}

	// 发布异步任务，失败时仅记录日志，不阻断任务创建
	// 资源存在性校验在异步策略中执行，任务最终状态可能为 FAILED
	if ts.taskExecutor == nil {
		ts.logger.Warn("任务执行器未初始化，任务将保持 PENDING 状态",
			zap.Int64("taskID", task.ID),
			zap.String("taskIDStr", taskIDStr))
	} else if err := ts.taskExecutor.PublishExportTask(ctx, task.TaskID, form); err != nil {
		ts.logger.Warn("发布导出任务失败，任务保持 PENDING 状态",
			zap.Int64("taskID", task.ID),
			zap.String("taskIDStr", taskIDStr),
			zap.Error(err))
	}

	ts.logger.Info("创建导出任务成功",
		zap.Int64("taskID", task.ID),
		zap.String("taskIDStr", taskIDStr),
		zap.String("type", form.Type),
		zap.Int64("userID", userID))

	return &task, nil
}

// GetTaskStatus 查询任务状态
// 任务不存在时返回 (nil, nil)，由调用方返回 data=null 给前端
func (ts *TaskService) GetTaskStatus(ctx context.Context, taskIDStr string) (*model.SysTask, error) {
	task, err := ts.getTaskFromCacheOrDB(ctx, taskIDStr)
	if err != nil {
		// 任务不存在时返回 nil，不返回错误
		if bizErr, ok := common.AsBizError(err); ok && bizErr.Code() == common.RESOURCE_NOT_FOUND {
			return nil, nil
		}
		return nil, err
	}
	return task, nil
}

// DownloadExportFile 下载导出文件
func (ts *TaskService) DownloadExportFile(ctx context.Context, taskIDStr string) (string, error) {
	task, err := ts.getTaskFromCacheOrDB(ctx, taskIDStr)
	if err != nil {
		return "", err
	}

	if task.Status != model.TaskStatusCompleted {
		ts.logger.Warn("任务未完成，无法下载", zap.String("taskID", taskIDStr), zap.String("status", string(task.Status)))
		return "", common.NewBizError(common.DATA_STATE_NOT_ALLOW, fmt.Sprintf("任务未完成，当前状态: %s", task.Status))
	}

	if task.IsExpired() {
		ts.logger.Warn("任务已过期，无法下载", zap.String("taskID", taskIDStr), zap.Time("expiresAt", *task.ExpiresAt))
		return "", common.NewBizError(common.DATA_STATE_NOT_ALLOW, "任务已过期")
	}

	if task.Result == "" {
		ts.logger.Warn("任务结果为空", zap.String("taskID", taskIDStr))
		return "", common.NewBizError(common.RESOURCE_NOT_FOUND, "任务结果为空")
	}

	downloadURL := task.Result
	ts.logger.Info("生成下载链接", zap.String("taskID", taskIDStr), zap.String("url", downloadURL))
	return downloadURL, nil
}

// CancelTask 取消导出任务
func (ts *TaskService) CancelTask(ctx context.Context, taskIDStr string, userID int64) error {
	task, err := ts.getTaskFromCacheOrDB(ctx, taskIDStr)
	if err != nil {
		return err
	}

	if task.CreateBy != userID {
		return common.NewBizError(common.OPERATION_NOT_ALLOW, "无权取消该任务")
	}

	if task.IsCompleted() {
		ts.logger.Warn("任务已完成或失败，无法取消", zap.String("taskID", taskIDStr), zap.String("status", string(task.Status)))
		return common.NewBizError(common.DATA_STATE_NOT_ALLOW, "任务已完成或失败，无法取消")
	}

	if task.Status == model.TaskStatusCancelled {
		ts.logger.Warn("任务已取消", zap.String("taskID", taskIDStr))
		return common.NewBizError(common.DATA_STATE_NOT_ALLOW, "任务已取消")
	}

	completedAt := time.Now()
	err = ts.taskRepo.UpdateFields(ctx, task.ID, map[string]interface{}{
		"status":       model.TaskStatusCancelled,
		"completed_at": &completedAt,
	})
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新任务状态失败", err)
	}

	// 同步更新内存中的 task 状态后再缓存，避免缓存旧状态
	task.Status = model.TaskStatusCancelled
	task.CompletedAt = &completedAt
	ts.cacheTask(ctx, task)

	cancelKey := TASK_CANCEL_PREFIX + taskIDStr
	err = ts.cache.Set(ctx, cancelKey, "true", TASK_CANCEL_EXPIRE_MS)
	if err != nil {
		ts.logger.Error("设置取消标志失败", zap.Error(err))
	}

	// WebSocket 推送任务取消
	ts.pushTaskWsMessage(taskIDStr, task.CreateBy, "task_status", map[string]interface{}{
		"status":        string(model.TaskStatusCancelled),
		"progress":      task.Progress,
		"result":        nil,
		"error_message": nil,
	})

	ts.logger.Info("取消导出任务成功", zap.String("taskID", taskIDStr))
	return nil
}

// getTaskFromCacheOrDB 从缓存或数据库获取任务
func (ts *TaskService) getTaskFromCacheOrDB(ctx context.Context, taskIDStr string) (*model.SysTask, error) {
	cacheKey := TASK_CACHE_PREFIX + taskIDStr

	cachedData, err := ts.cache.Get(ctx, cacheKey)
	if err == nil && cachedData != "" {
		var task model.SysTask
		if err := json.Unmarshal([]byte(cachedData), &task); err == nil {
			return &task, nil
		}
	}

	task, err := ts.taskRepo.FindByTaskID(ctx, taskIDStr)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询任务失败", err)
	}
	if task == nil {
		ts.logger.Warn("任务不存在", zap.String("taskID", taskIDStr))
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "任务不存在")
	}

	ts.cacheTask(ctx, task)
	return task, nil
}

// cacheTask 缓存任务信息到Redis
func (ts *TaskService) cacheTask(ctx context.Context, task *model.SysTask) {
	cacheKey := TASK_CACHE_PREFIX + task.TaskID
	taskJSON, err := json.Marshal(task)
	if err != nil {
		ts.logger.Error("任务序列化失败", zap.Error(err))
		return
	}

	err = ts.cache.Set(ctx, cacheKey, string(taskJSON), TASK_EXPIRE_HOURS)
	if err != nil {
		ts.logger.Error("缓存任务失败", zap.Error(err))
	}
}

// ConvertToTaskVO 转换为任务VO
func (ts *TaskService) ConvertToTaskVO(task *model.SysTask) *vo.TaskVO {
	result := &vo.TaskVO{
		TaskID:         task.TaskID,
		Status:         string(task.Status),
		Progress:       task.Progress,
		TotalFiles:     task.TotalFiles,
		ProcessedFiles: task.ProcessedFiles,
		ExpiresAt:      task.ExpiresAt,
		CreatedAt:      task.CreatedAt,
		StartedAt:      task.StartedAt,
		CompletedAt:    task.CompletedAt,
		Error:          task.ErrorMessage,
		IdempotencyKey: task.IdempotencyKey,
		RetryCount:     task.RetryCount,
		WorkerID:       task.WorkerID,
	}

	if task.Status == model.TaskStatusCompleted && task.Result != "" {
		result.DownloadURL = task.Result
	}

	return result
}

// RetryTask 重试失败的任务（重放入口）
// 仅 FAILED 状态的任务可重试，重置状态后重新发布到 MQ
func (ts *TaskService) RetryTask(ctx context.Context, taskIDStr string, userID int64) (*model.SysTask, error) {
	task, err := ts.getTaskFromCacheOrDB(ctx, taskIDStr)
	if err != nil {
		return nil, err
	}

	if task.CreateBy != userID {
		return nil, common.NewBizError(common.OPERATION_NOT_ALLOW, "无权重试该任务")
	}

	if task.Status != model.TaskStatusFailed {
		return nil, common.NewBizError(common.DATA_STATE_NOT_ALLOW, "仅失败的任务可重试")
	}

	// 重置任务状态为 PENDING，清除错误信息
	now := time.Now()
	expiresAt := now.Add(TASK_EXPIRE_HOURS)
	err = ts.taskRepo.UpdateFields(ctx, task.ID, map[string]interface{}{
		"status":        model.TaskStatusPending,
		"progress":      0,
		"processed_files": 0,
		"error_message": "",
		"started_at":    nil,
		"completed_at":  nil,
		"expires_at":    &expiresAt,
	})
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "重置任务状态失败", err)
	}

	// 同步内存中的 task 对象
	task.Status = model.TaskStatusPending
	task.Progress = 0
	task.ProcessedFiles = 0
	task.ErrorMessage = ""
	task.StartedAt = nil
	task.CompletedAt = nil
	task.ExpiresAt = &expiresAt
	ts.cacheTask(ctx, task)

	// 重新发布到 MQ
	var form bo.ExportTaskCreateForm
	if err := json.Unmarshal([]byte(task.Params), &form); err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "解析任务参数失败", err)
	}

	if ts.taskExecutor != nil {
		if err := ts.taskExecutor.PublishExportTask(ctx, task.TaskID, form); err != nil {
			ts.logger.Warn("重试发布任务失败，任务保持 PENDING 状态",
				zap.String("taskID", taskIDStr), zap.Error(err))
		}
	}

	// WebSocket 推送重试
	ts.pushTaskWsMessage(taskIDStr, task.CreateBy, "task_status", map[string]interface{}{
		"status":        string(model.TaskStatusPending),
		"progress":      0,
		"result":        nil,
		"error_message": nil,
	})

	ts.logger.Info("重试任务成功", zap.String("taskID", taskIDStr))
	return task, nil
}

// UpdateTaskStatus 更新任务状态（供 Consumer 调用）
// ctx 用于携带用户身份（userID），保证审计字段 update_by 正确填充
func (ts *TaskService) UpdateTaskStatus(ctx context.Context, taskIDStr string, status model.TaskStatus, errorMessage string) error {
	task, err := ts.taskRepo.FindByTaskID(ctx, taskIDStr)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询任务失败", err)
	}
	if task == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "任务不存在")
	}

	fields := map[string]interface{}{
		"status": status,
	}
	now := time.Now()
	switch status {
	case model.TaskStatusProcessing:
		fields["started_at"] = &now
	case model.TaskStatusCompleted:
		fields["progress"] = 100
		fields["completed_at"] = &now
		expiresAt := now.Add(7 * 24 * time.Hour)
		fields["expires_at"] = &expiresAt
	case model.TaskStatusFailed:
		fields["error_message"] = errorMessage
		fields["completed_at"] = &now
	case model.TaskStatusCancelled:
		fields["completed_at"] = &now
	}

	if err := ts.taskRepo.UpdateFields(ctx, task.ID, fields); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新任务状态失败", err)
	}

	// 同步内存对象并更新缓存
	task.Status = status
	if status == model.TaskStatusProcessing {
		task.StartedAt = &now
	}
	if status == model.TaskStatusCompleted {
		task.Progress = 100
		task.CompletedAt = &now
	}
	if status == model.TaskStatusFailed {
		task.ErrorMessage = errorMessage
		task.CompletedAt = &now
	}
	if status == model.TaskStatusCancelled {
		task.CompletedAt = &now
	}
	ts.cacheTask(ctx, task)

	// WebSocket 推送状态变更
	ts.pushTaskWsMessage(taskIDStr, task.CreateBy, "task_status", map[string]interface{}{
		"status":        string(status),
		"progress":      task.Progress,
		"result":        task.Result,
		"error_message": errorMessage,
	})

	return nil
}

// UpdateTaskProgress 更新任务进度（供 Consumer 调用）
// ctx 用于携带用户身份（userID），保证审计字段 update_by 正确填充
func (ts *TaskService) UpdateTaskProgress(ctx context.Context, taskIDStr string, progress, current, total int) error {
	task, err := ts.taskRepo.FindByTaskID(ctx, taskIDStr)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询任务失败", err)
	}
	if task == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "任务不存在")
	}

	if err := ts.taskRepo.UpdateFields(ctx, task.ID, map[string]interface{}{
		"progress":        progress,
		"processed_files": current,
		"total_files":     total,
	}); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新任务进度失败", err)
	}

	task.Progress = progress
	task.ProcessedFiles = current
	task.TotalFiles = total
	ts.cacheTask(ctx, task)

	// WebSocket 推送进度
	ts.pushTaskWsMessage(taskIDStr, task.CreateBy, "task_progress", map[string]interface{}{
		"progress":        progress,
		"status":          string(model.TaskStatusProcessing),
		"processed_files": current,
		"total_files":     total,
	})

	return nil
}

// UpdateRetryCount 更新 MQ 重试次数
// ctx 用于携带用户身份（userID），保证审计字段 update_by 正确填充
func (ts *TaskService) UpdateRetryCount(ctx context.Context, taskIDStr string, retryCount int) error {
	task, err := ts.taskRepo.FindByTaskID(ctx, taskIDStr)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询任务失败", err)
	}
	if task == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "任务不存在")
	}

	if err := ts.taskRepo.UpdateFields(ctx, task.ID, map[string]interface{}{
		"retry_count": retryCount,
	}); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新重试次数失败", err)
	}

	task.RetryCount = retryCount
	ts.cacheTask(ctx, task)
	return nil
}

// pushTaskWsMessage 通过 WebSocket 推送任务消息（Redis Pub/Sub 跨实例投递，对齐 Python 消息格式）
func (ts *TaskService) pushTaskWsMessage(taskIDStr string, createdBy int64, msgType string, data map[string]interface{}) {
	manager := websocket.GetManager()
	if manager == nil {
		ts.logger.Debug("WebSocket 管理器未初始化，跳过推送",
			zap.String("taskID", taskIDStr), zap.String("type", msgType))
		return
	}

	// 构建完整消息（对齐 Python/Java 消息格式）
	message := map[string]interface{}{
		"type":      msgType,
		"task_id":   taskIDStr,
		"timestamp": time.Now().Format("2006-01-02T15:04:05.000000"),
	}
	for k, v := range data {
		message[k] = v
	}

	manager.PublishToUser(createdBy, message)
}

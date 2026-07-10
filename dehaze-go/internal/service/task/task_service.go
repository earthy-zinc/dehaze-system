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
	"github.com/google/uuid"
	"go.uber.org/zap"
)

const (
	TASK_CACHE_PREFIX     = "export:task:"
	TASK_CANCEL_PREFIX    = "task:cancel:"
	TASK_EXPIRE_HOURS     = 24 * time.Hour
	TASK_CANCEL_EXPIRE_MS = 5 * time.Minute
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
func (ts *TaskService) CreateExportTask(form bo.ExportTaskCreateForm, userID int64) (*model.SysTask, error) {
	ctx := context.Background()
	taskIDStr := uuid.New().String()

	paramsJSON, err := json.Marshal(form)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "参数序列化失败", err)
	}

	now := time.Now()
	expiresAt := now.Add(TASK_EXPIRE_HOURS)

	task := model.SysTask{
		TaskID:         taskIDStr,
		TaskType:       model.TaskTypeExport,
		Status:         model.TaskStatusPending,
		Progress:       0,
		TotalFiles:     0,
		ProcessedFiles: 0,
		Params:         string(paramsJSON),
		CreatedBy:      userID,
		ExpiresAt:      &expiresAt,
	}

	err = ts.taskRepo.Create(ctx, &task)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "保存任务失败", err)
	}

	ts.cacheTask(&task)
	if ts.taskExecutor == nil {
		return nil, common.NewBizError(common.SYSTEM_EXECUTION_ERROR, "任务执行器未初始化")
	}
	if err := ts.taskExecutor.PublishExportTask(ctx, task.ID, form); err != nil {
		completedAt := time.Now()
		_ = ts.taskRepo.UpdateFields(ctx, task.ID, map[string]interface{}{
			"status":        model.TaskStatusFailed,
			"error_message": err.Error(),
			"completed_at":  &completedAt,
		})
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "发布导出任务失败", err)
	}

	ts.logger.Info("创建导出任务成功",
		zap.Int64("taskID", task.ID),
		zap.String("taskIDStr", taskIDStr),
		zap.String("type", form.Type),
		zap.Int64("userID", userID))

	return &task, nil
}

// GetTaskStatus 查询任务状态
func (ts *TaskService) GetTaskStatus(taskIDStr string) (*model.SysTask, error) {
	task, err := ts.getTaskFromCacheOrDB(taskIDStr)
	if err != nil {
		return nil, err
	}
	return task, nil
}

// DownloadExportFile 下载导出文件
func (ts *TaskService) DownloadExportFile(taskIDStr string) (string, error) {
	task, err := ts.getTaskFromCacheOrDB(taskIDStr)
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
func (ts *TaskService) CancelTask(taskIDStr string, userID int64) error {
	ctx := context.Background()
	task, err := ts.getTaskFromCacheOrDB(taskIDStr)
	if err != nil {
		return err
	}

	if task.CreatedBy != userID {
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

	ts.cacheTask(task)

	cancelKey := TASK_CANCEL_PREFIX + taskIDStr
	err = ts.cache.Set(ctx, cancelKey, "true", TASK_CANCEL_EXPIRE_MS)
	if err != nil {
		ts.logger.Error("设置取消标志失败", zap.Error(err))
	}

	ts.logger.Info("取消导出任务成功", zap.String("taskID", taskIDStr))
	return nil
}

// getTaskFromCacheOrDB 从缓存或数据库获取任务
func (ts *TaskService) getTaskFromCacheOrDB(taskIDStr string) (*model.SysTask, error) {
	ctx := context.Background()
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

	ts.cacheTask(task)
	return task, nil
}

// cacheTask 缓存任务信息到Redis
func (ts *TaskService) cacheTask(task *model.SysTask) {
	cacheKey := TASK_CACHE_PREFIX + task.TaskID
	taskJSON, err := json.Marshal(task)
	if err != nil {
		ts.logger.Error("任务序列化失败", zap.Error(err))
		return
	}

	ctx := context.Background()
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
	}

	if task.Status == model.TaskStatusCompleted && task.Result != "" {
		result.DownloadURL = task.Result
	}

	return result
}

// CleanupExpiredTasks 清理过期任务
func (ts *TaskService) CleanupExpiredTasks() error {
	ctx := context.Background()
	threshold := time.Now().Add(-TASK_EXPIRE_HOURS)

	affected, err := ts.taskRepo.UpdateExpiredTasks(ctx, threshold)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "清理过期任务失败", err)
	}

	ts.logger.Info("清理过期任务完成", zap.Int64("affected", affected))
	return nil
}

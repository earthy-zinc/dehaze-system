package service

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/executor"
	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/google/uuid"
	"go.uber.org/zap"
)

const (
	TASK_CACHE_PREFIX     = "export:task:"
	TASK_CANCEL_PREFIX    = "task:cancel:"
	TASK_EXPIRE_HOURS     = 24 * time.Hour
	TASK_CANCEL_EXPIRE_MS = 5 * time.Minute
)

// TaskExecutor 异步任务执行器
type TaskExecutor struct {
	workerPools map[string]*executor.WorkerPool
	initialized bool
	mu          sync.RWMutex
	taskRepo    repository.ITaskRepository
	datasetRepo repository.IDatasetRepository
	cache       types.ICache
	logger      *zap.Logger
	config      *options.AsyncTask
}

// NewTaskExecutor 创建任务执行器
func NewTaskExecutor(
	taskRepo repository.ITaskRepository,
	datasetRepo repository.IDatasetRepository,
	c types.ICache,
	logger *zap.Logger,
	cfg *options.AsyncTask,
) *TaskExecutor {
	return &TaskExecutor{
		workerPools: make(map[string]*executor.WorkerPool),
		taskRepo:    taskRepo,
		datasetRepo: datasetRepo,
		cache:       c,
		logger:      logger,
		config:      cfg,
	}
}

// Initialize 初始化Worker池
func (te *TaskExecutor) Initialize() {
	te.mu.Lock()
	defer te.mu.Unlock()

	if te.initialized {
		return
	}

	exportPool := executor.NewWorkerPool("export-worker", te.config.ExportTask)
	exportPool.Start()
	te.workerPools["export"] = exportPool

	datasetPool := executor.NewWorkerPool("dataset-worker", te.config.DatasetTask)
	datasetPool.Start()
	te.workerPools["dataset"] = datasetPool

	thumbnailPool := executor.NewWorkerPool("thumbnail-worker", te.config.ImageTask)
	thumbnailPool.Start()
	te.workerPools["thumbnail"] = thumbnailPool

	te.logger.Info("TaskExecutor初始化完成",
		zap.Int("exportWorkers", int(exportPool.GetStatus()["workerCount"].(float64))),
		zap.Int("datasetWorkers", int(datasetPool.GetStatus()["workerCount"].(float64))),
		zap.Int("thumbnailWorkers", int(thumbnailPool.GetStatus()["workerCount"].(float64))),
	)

	te.initialized = true
}

// Shutdown 关闭所有Worker池
func (te *TaskExecutor) Shutdown() {
	te.mu.RLock()
	defer te.mu.RUnlock()

	if !te.initialized {
		return
	}

	for name, pool := range te.workerPools {
		te.logger.Info("关闭Worker池", zap.String("pool", name))
		pool.Stop()
	}

	te.initialized = false
}

// SubmitExportTask 提交导出任务
func (te *TaskExecutor) SubmitExportTask(taskID int64, form bo.ExportTaskCreateForm) {
	pool := te.getWorkerPool("export")
	if pool == nil {
		te.logger.Error("导出任务Worker池不存在")
		te.updateTaskStatusToFailed(taskID, "Worker池未初始化")
		return
	}

	taskIDStr := fmt.Sprintf("%d", taskID)
	totalFiles := te.calculateTotalFiles(form)

	taskCtx, err := pool.SubmitWithCtx(taskIDStr, "export", totalFiles, func(ctx context.Context) error {
		return te.executeExportTask(ctx, taskID, form)
	})

	if err != nil {
		te.logger.Error("提交导出任务失败", zap.Int64("taskID", taskID), zap.Error(err))
		te.updateTaskStatusToFailed(taskID, err.Error())
		return
	}

	go te.monitorTaskProgress(taskID, taskCtx.ProgressChan, taskIDStr)
}

// executeExportTask 执行导出任务
func (te *TaskExecutor) executeExportTask(ctx context.Context, taskID int64, form bo.ExportTaskCreateForm) error {
	task, err := te.taskRepo.FindByID(ctx, taskID)
	if err != nil {
		return fmt.Errorf("任务不存在: %w", err)
	}
	if task == nil {
		return fmt.Errorf("任务不存在: id=%d", taskID)
	}

	startedAt := time.Now()
	err = te.taskRepo.UpdateFields(ctx, taskID, map[string]interface{}{
		"status":          model.TaskStatusProcessing,
		"started_at":      &startedAt,
		"total_files":     te.calculateTotalFiles(form),
		"processed_files": 0,
	})
	if err != nil {
		return fmt.Errorf("更新任务状态失败: %w", err)
	}

	if te.isTaskCanceled(ctx, fmt.Sprintf("%d", taskID)) {
		return te.cancelTask(ctx, taskID, "任务被取消")
	}

	exportDir := filepath.Join(os.TempDir(), "exports", fmt.Sprintf("task_%d", taskID))
	if err := os.MkdirAll(exportDir, 0755); err != nil {
		return fmt.Errorf("创建导出目录失败: %w", err)
	}
	defer os.RemoveAll(exportDir)

	var zipPath string
	switch form.Type {
	case "dataset":
		zipPath, err = te.exportDataset(ctx, taskID, form, exportDir)
	case "dataset_item", "batch_items":
		zipPath, err = te.exportDatasetItems(ctx, taskID, form, exportDir)
	case "custom":
		zipPath, err = te.exportCustom(ctx, taskID, form, exportDir)
	default:
		return fmt.Errorf("不支持的导出类型: %s", form.Type)
	}

	if err != nil {
		return err
	}

	if te.isTaskCanceled(ctx, fmt.Sprintf("%d", taskID)) {
		return te.cancelTask(ctx, taskID, "任务被取消")
	}

	downloadURL, err := te.uploadExportFile(zipPath, taskID)
	if err != nil {
		return fmt.Errorf("上传导出文件失败: %w", err)
	}

	completedAt := time.Now()
	expiresAt := time.Now().Add(TASK_EXPIRE_HOURS)

	err = te.taskRepo.UpdateFields(ctx, taskID, map[string]interface{}{
		"status":       model.TaskStatusCompleted,
		"progress":     100,
		"result":       downloadURL,
		"completed_at": &completedAt,
		"expires_at":   &expiresAt,
	})

	if err != nil {
		return fmt.Errorf("更新任务状态失败: %w", err)
	}

	te.logger.Info("导出任务完成", zap.Int64("taskID", taskID), zap.String("downloadURL", downloadURL))
	return nil
}

// exportDataset 导出数据集
func (te *TaskExecutor) exportDataset(ctx context.Context, taskID int64, form bo.ExportTaskCreateForm, exportDir string) (string, error) {
	datasetID := form.TargetID

	dataset, err := te.datasetRepo.FindByID(ctx, datasetID)
	if err != nil {
		return "", fmt.Errorf("查询数据集失败: %w", err)
	}
	if dataset == nil {
		return "", fmt.Errorf("数据集不存在: id=%d", datasetID)
	}

	zipFileName := fmt.Sprintf("%s_export_%s.zip", dataset.Name, time.Now().Format("20060102_150405"))
	zipPath := filepath.Join(exportDir, zipFileName)

	zipFile, err := os.Create(zipPath)
	if err != nil {
		return "", fmt.Errorf("创建ZIP文件失败: %w", err)
	}
	defer zipFile.Close()

	te.logger.Info("创建导出ZIP", zap.Int64("taskID", taskID), zap.String("zipPath", zipPath))
	return zipPath, nil
}

// exportDatasetItems 导出数据集项
func (te *TaskExecutor) exportDatasetItems(ctx context.Context, taskID int64, form bo.ExportTaskCreateForm, exportDir string) (string, error) {
	zipFileName := fmt.Sprintf("items_export_%s.zip", time.Now().Format("20060102_150405"))
	zipPath := filepath.Join(exportDir, zipFileName)

	zipFile, err := os.Create(zipPath)
	if err != nil {
		return "", fmt.Errorf("创建ZIP文件失败: %w", err)
	}
	defer zipFile.Close()

	te.logger.Info("创建导出ZIP", zap.Int64("taskID", taskID), zap.String("zipPath", zipPath))
	return zipPath, nil
}

// exportCustom 自定义导出
func (te *TaskExecutor) exportCustom(ctx context.Context, taskID int64, form bo.ExportTaskCreateForm, exportDir string) (string, error) {
	zipFileName := fmt.Sprintf("custom_export_%s.zip", time.Now().Format("20060102_150405"))
	zipPath := filepath.Join(exportDir, zipFileName)

	zipFile, err := os.Create(zipPath)
	if err != nil {
		return "", fmt.Errorf("创建ZIP文件失败: %w", err)
	}
	defer zipFile.Close()

	return zipPath, nil
}

// uploadExportFile 上传导出文件到对象存储
func (te *TaskExecutor) uploadExportFile(zipPath string, taskID int64) (string, error) {
	return fmt.Sprintf("/api/download/export/%d", taskID), nil
}

// isTaskCanceled 检查任务是否被取消
func (te *TaskExecutor) isTaskCanceled(ctx context.Context, taskID string) bool {
	select {
	case <-ctx.Done():
		return true
	default:
		cancelKey := TASK_CANCEL_PREFIX + taskID
		canceled, _ := te.cache.Get(ctx, cancelKey)
		return canceled == "true"
	}
}

// cancelTask 取消任务
func (te *TaskExecutor) cancelTask(ctx context.Context, taskID int64, reason string) error {
	completedAt := time.Now()
	err := te.taskRepo.UpdateFields(ctx, taskID, map[string]interface{}{
		"status":        model.TaskStatusCancelled,
		"error_message": reason,
		"completed_at":  &completedAt,
		"progress":      0,
	})

	if err != nil {
		return fmt.Errorf("更新任务状态失败: %w", err)
	}

	te.logger.Warn("任务已取消", zap.Int64("taskID", taskID), zap.String("reason", reason))
	return nil
}

// updateTaskStatusToFailed 更新任务状态为失败
func (te *TaskExecutor) updateTaskStatusToFailed(taskID int64, errorMsg string) {
	ctx := context.Background()
	completedAt := time.Now()
	te.taskRepo.UpdateFields(ctx, taskID, map[string]interface{}{
		"status":        model.TaskStatusFailed,
		"error_message": errorMsg,
		"completed_at":  &completedAt,
	})
}

// monitorTaskProgress 监听任务进度
func (te *TaskExecutor) monitorTaskProgress(taskID int64, progressChan <-chan int, taskIDStr string) {
	ctx := context.Background()
	for progress := range progressChan {
		err := te.taskRepo.UpdateFields(ctx, taskID, map[string]interface{}{"progress": progress})
		if err != nil {
			te.logger.Error("更新任务进度失败", zap.Int64("taskID", taskID), zap.Int("progress", progress), zap.Error(err))
		}

		cacheKey := TASK_CACHE_PREFIX + taskIDStr
		cachedData, err := te.cache.Get(ctx, cacheKey)
		if err == nil && cachedData != "" {
			var cachedTask model.SysTask
			if json.Unmarshal([]byte(cachedData), &cachedTask) == nil {
				cachedTask.Progress = progress
				if taskJSON, marshalErr := json.Marshal(&cachedTask); marshalErr == nil {
					te.cache.Set(ctx, cacheKey, string(taskJSON), TASK_EXPIRE_HOURS)
				}
			}
		}
	}
}

// getWorkerPool 获取指定类型的Worker池
func (te *TaskExecutor) getWorkerPool(poolType string) *executor.WorkerPool {
	te.mu.RLock()
	defer te.mu.RUnlock()
	return te.workerPools[poolType]
}

// calculateTotalFiles 计算需要处理的文件总数
func (te *TaskExecutor) calculateTotalFiles(form bo.ExportTaskCreateForm) int {
	ctx := context.Background()
	switch form.Type {
	case "dataset":
		count, _ := te.taskRepo.CountDatasetItems(ctx, form.TargetID)
		return int(count)
	case "dataset_item", "batch_items":
		count, _ := te.taskRepo.CountItemFiles(ctx, form.TargetIDs)
		return int(count)
	default:
		return 0
	}
}

// TaskService 任务服务
type TaskService struct {
	taskRepo     repository.ITaskRepository
	datasetRepo  repository.IDatasetRepository
	cache        types.ICache
	logger       *zap.Logger
	taskExecutor *TaskExecutor
}

// NewTaskService 创建任务服务
func NewTaskService(
	taskRepo repository.ITaskRepository,
	datasetRepo repository.IDatasetRepository,
	c types.ICache,
	logger *zap.Logger,
	cfg *options.AsyncTask,
) *TaskService {
	taskExecutor := NewTaskExecutor(taskRepo, datasetRepo, c, logger, cfg)
	taskExecutor.Initialize()

	return &TaskService{
		taskRepo:     taskRepo,
		datasetRepo:  datasetRepo,
		cache:        c,
		logger:       logger,
		taskExecutor: taskExecutor,
	}
}

// GetTaskExecutor 获取任务执行器（用于关闭）
func (ts *TaskService) GetTaskExecutor() *TaskExecutor {
	return ts.taskExecutor
}

// CreateExportTask 创建导出任务
func (ts *TaskService) CreateExportTask(form bo.ExportTaskCreateForm, userID int64) (*model.SysTask, error) {
	ctx := context.Background()
	taskIDStr := uuid.New().String()

	paramsJSON, err := json.Marshal(form)
	if err != nil {
		return nil, fmt.Errorf("参数序列化失败: %w", err)
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
		return nil, fmt.Errorf("保存任务失败: %w", err)
	}

	ts.cacheTask(&task)
	go ts.taskExecutor.SubmitExportTask(task.ID, form)

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
		return "", fmt.Errorf("任务未完成，当前状态: %s", task.Status)
	}

	if task.IsExpired() {
		ts.logger.Warn("任务已过期，无法下载", zap.String("taskID", taskIDStr), zap.Time("expiresAt", *task.ExpiresAt))
		return "", fmt.Errorf("任务已过期")
	}

	if task.Result == "" {
		ts.logger.Warn("任务结果为空", zap.String("taskID", taskIDStr))
		return "", fmt.Errorf("任务结果为空")
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
		return fmt.Errorf("无权取消该任务")
	}

	if task.IsCompleted() {
		ts.logger.Warn("任务已完成或失败，无法取消", zap.String("taskID", taskIDStr), zap.String("status", string(task.Status)))
		return fmt.Errorf("任务已完成或失败，无法取消")
	}

	if task.Status == model.TaskStatusCancelled {
		ts.logger.Warn("任务已取消", zap.String("taskID", taskIDStr))
		return fmt.Errorf("任务已取消")
	}

	completedAt := time.Now()
	err = ts.taskRepo.UpdateFields(ctx, task.ID, map[string]interface{}{
		"status":       model.TaskStatusCancelled,
		"completed_at": &completedAt,
	})
	if err != nil {
		return fmt.Errorf("更新任务状态失败: %w", err)
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
		return nil, fmt.Errorf("查询任务失败: %w", err)
	}
	if task == nil {
		ts.logger.Warn("任务不存在", zap.String("taskID", taskIDStr))
		return nil, fmt.Errorf("任务不存在")
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
		return fmt.Errorf("清理过期任务失败: %w", err)
	}

	ts.logger.Info("清理过期任务完成", zap.Int64("affected", affected))
	return nil
}

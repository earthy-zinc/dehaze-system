package job

import (
	evalrepo "github.com/earthyzinc/dehaze-go/internal/repository/eval_log"
	predrepo "github.com/earthyzinc/dehaze-go/internal/repository/pred_log"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
)

// JobManager 定时任务管理器
type JobManager struct {
	cleanupJob *CleanupJob
}

// NewJobManager 创建任务管理器
func NewJobManager(storageSvc StorageService, predLogRepo predrepo.IPredLogRepository, evalLogRepo evalrepo.IEvalLogRepository) *JobManager {
	return &JobManager{
		cleanupJob: NewCleanupJob(storageSvc, predLogRepo, evalLogRepo),
	}
}

// Start 启动所有定时任务
func (m *JobManager) Start() {
	logger.Info("启动定时任务管理器")
	m.cleanupJob.Start()
}

// Stop 停止所有定时任务
func (m *JobManager) Stop() {
	logger.Info("停止定时任务管理器")
	m.cleanupJob.Stop()
}

// GetCleanupJob 获取清理任务实例（用于手动触发）
func (m *JobManager) GetCleanupJob() *CleanupJob {
	return m.cleanupJob
}

// 全局任务管理器实例
var jobManager *JobManager

// InitJobs 初始化定时任务
func InitJobs(storageSvc StorageService, predLogRepo predrepo.IPredLogRepository, evalLogRepo evalrepo.IEvalLogRepository) {
	jobManager = NewJobManager(storageSvc, predLogRepo, evalLogRepo)
	jobManager.Start()
	logger.Info("定时任务初始化完成")
}

// StopJobs 停止定时任务
func StopJobs() {
	if jobManager != nil {
		jobManager.Stop()
		logger.Info("定时任务已停止")
	}
}

// GetJobManager 获取任务管理器
func GetJobManager() *JobManager {
	return jobManager
}

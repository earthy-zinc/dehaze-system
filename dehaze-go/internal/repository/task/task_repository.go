package task

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
	"gorm.io/gorm"
)

type taskRepository struct {
	db *gorm.DB
}

func NewTaskRepository(db *gorm.DB) ITaskRepository {
	return &taskRepository{db: db}
}

func (r *taskRepository) FindByID(ctx context.Context, id int64) (*model.SysTask, error) {
	var task model.SysTask
	err := r.db.WithContext(ctx).Where("id = ?", id).First(&task).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &task, err
}

func (r *taskRepository) FindByTaskID(ctx context.Context, taskID string) (*model.SysTask, error) {
	var task model.SysTask
	err := r.db.WithContext(ctx).Where("task_id = ?", taskID).First(&task).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &task, err
}

func (r *taskRepository) FindPage(ctx context.Context, q any) (*read.PageResult[read.Task], error) {
	// 根据实际查询条件实现分页
	return nil, nil
}

func (r *taskRepository) Create(ctx context.Context, task *model.SysTask) error {
	return r.db.WithContext(ctx).Create(task).Error
}

func (r *taskRepository) Update(ctx context.Context, task *model.SysTask) error {
	return r.db.WithContext(ctx).Save(task).Error
}

func (r *taskRepository) UpdateFields(ctx context.Context, id int64, fields map[string]interface{}) error {
	return r.db.WithContext(ctx).Model(&model.SysTask{}).Where("id = ?", id).Updates(fields).Error
}

func (r *taskRepository) UpdateStatus(ctx context.Context, id int64, status int8) error {
	return r.db.WithContext(ctx).Model(&model.SysTask{}).Where("id = ?", id).Update("status", status).Error
}

func (r *taskRepository) Delete(ctx context.Context, ids []int64) error {
	return r.db.WithContext(ctx).Where("id IN ?", ids).Delete(&model.SysTask{}).Error
}

func (r *taskRepository) UpdateExpiredTasks(ctx context.Context, threshold time.Time) (int64, error) {
	result := r.db.WithContext(ctx).Model(&model.SysTask{}).
		Where("expires_at < ?", threshold).
		Updates(map[string]interface{}{
			"status":        model.TaskStatusCancelled,
			"error_message": "任务已过期",
		})
	return result.RowsAffected, result.Error
}

func (r *taskRepository) CountDatasetItems(ctx context.Context, datasetID int64) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysDatasetItem{}).Where("dataset_id = ?", datasetID).Count(&count).Error
	return count, err
}

func (r *taskRepository) CountItemFiles(ctx context.Context, itemIDs []int64) (int64, error) {
	if len(itemIDs) == 0 {
		return 0, nil
	}
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysItemFile{}).Where("item_id IN ?", itemIDs).Count(&count).Error
	return count, err
}

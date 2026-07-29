package task

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
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
	if err != nil {
		return nil, err
	}
	return &task, nil
}

func (r *taskRepository) FindByTaskID(ctx context.Context, taskID string) (*model.SysTask, error) {
	var task model.SysTask
	err := r.db.WithContext(ctx).Where("task_id = ?", taskID).First(&task).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &task, nil
}

func (r *taskRepository) FindByIdempotencyKey(ctx context.Context, key string) (*model.SysTask, error) {
	var task model.SysTask
	err := r.db.WithContext(ctx).Where("idempotency_key = ?", key).First(&task).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &task, nil
}

func (r *taskRepository) FindPage(ctx context.Context, q *query.TaskPageQuery) (*read.PageResult[read.Task], error) {
	pageNum := q.PageNum
	pageSize := q.PageSize
	if pageNum <= 0 {
		pageNum = 1
	}
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).Model(&model.SysTask{})
	if q.Status != nil {
		db = db.Where("status = ?", *q.Status)
	}
	if q.TaskType != "" {
		db = db.Where("task_type = ?", q.TaskType)
	}
	if q.TaskCategory != "" {
		switch q.TaskCategory {
		case "export":
			db = db.Where("task_type LIKE ?", "%\\_export")
		case "import":
			db = db.Where("task_type LIKE ?", "%\\_import")
		}
	}
	if q.UserID > 0 {
		db = db.Where("create_by = ?", q.UserID)
	}

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, err
	}

	var tasks []model.SysTask
	if err := db.Order("create_time DESC").
		Offset((pageNum - 1) * pageSize).
		Limit(pageSize).
		Find(&tasks).Error; err != nil {
		return nil, err
	}

	list := make([]read.Task, 0, len(tasks))
	for i := range tasks {
		t := &tasks[i]
		item := read.Task{
			TaskID:         t.TaskID,
			TaskType:       string(t.TaskType),
			Status:         int8(t.Status),
			Progress:       t.Progress,
			TotalFiles:     t.TotalFiles,
			ProcessedFiles: t.ProcessedFiles,
			ExpiresAt:      t.ExpiresAt,
			CreatedAt:      t.CreatedAt,
			StartedAt:      t.StartedAt,
			CompletedAt:   t.CompletedAt,
			Error:          t.ErrorMessage,
		}
		if t.Status == model.TaskStatusCompleted && t.Result != "" {
			item.DownloadURL = t.Result
		}
		list = append(list, item)
	}

	return &read.PageResult[read.Task]{
		List:     list,
		Total:    total,
		PageNum:  pageNum,
		PageSize: pageSize,
	}, nil
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
	return r.db.WithContext(ctx).Model(&model.SysTask{}).Where("id = ?", id).
		Updates(map[string]interface{}{"status": status}).Error
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

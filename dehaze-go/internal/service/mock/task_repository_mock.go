package mock

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

type TaskRepositoryMock struct {
	FindByIDFunc           func(ctx context.Context, id int64) (*model.SysTask, error)
	FindByTaskIDFunc       func(ctx context.Context, taskID string) (*model.SysTask, error)
	FindPageFunc           func(ctx context.Context, q any) (*vo.PageResult[vo.TaskVO], error)
	CreateFunc             func(ctx context.Context, task *model.SysTask) error
	UpdateFunc             func(ctx context.Context, task *model.SysTask) error
	UpdateFieldsFunc       func(ctx context.Context, id int64, fields map[string]interface{}) error
	UpdateStatusFunc       func(ctx context.Context, id int64, status int8) error
	DeleteFunc             func(ctx context.Context, ids []int64) error
	UpdateExpiredTasksFunc func(ctx context.Context, threshold time.Time) (int64, error)
	CountDatasetItemsFunc  func(ctx context.Context, datasetID int64) (int64, error)
	CountItemFilesFunc     func(ctx context.Context, itemIDs []int64) (int64, error)
}

func (m *TaskRepositoryMock) FindByID(ctx context.Context, id int64) (*model.SysTask, error) {
	if m.FindByIDFunc != nil {
		return m.FindByIDFunc(ctx, id)
	}
	return nil, nil
}

func (m *TaskRepositoryMock) FindByTaskID(ctx context.Context, taskID string) (*model.SysTask, error) {
	if m.FindByTaskIDFunc != nil {
		return m.FindByTaskIDFunc(ctx, taskID)
	}
	return nil, nil
}

func (m *TaskRepositoryMock) FindPage(ctx context.Context, q any) (*vo.PageResult[vo.TaskVO], error) {
	if m.FindPageFunc != nil {
		return m.FindPageFunc(ctx, q)
	}
	return nil, nil
}

func (m *TaskRepositoryMock) Create(ctx context.Context, task *model.SysTask) error {
	if m.CreateFunc != nil {
		return m.CreateFunc(ctx, task)
	}
	return nil
}

func (m *TaskRepositoryMock) Update(ctx context.Context, task *model.SysTask) error {
	if m.UpdateFunc != nil {
		return m.UpdateFunc(ctx, task)
	}
	return nil
}

func (m *TaskRepositoryMock) UpdateFields(ctx context.Context, id int64, fields map[string]interface{}) error {
	if m.UpdateFieldsFunc != nil {
		return m.UpdateFieldsFunc(ctx, id, fields)
	}
	return nil
}

func (m *TaskRepositoryMock) UpdateStatus(ctx context.Context, id int64, status int8) error {
	if m.UpdateStatusFunc != nil {
		return m.UpdateStatusFunc(ctx, id, status)
	}
	return nil
}

func (m *TaskRepositoryMock) Delete(ctx context.Context, ids []int64) error {
	if m.DeleteFunc != nil {
		return m.DeleteFunc(ctx, ids)
	}
	return nil
}

func (m *TaskRepositoryMock) UpdateExpiredTasks(ctx context.Context, threshold time.Time) (int64, error) {
	if m.UpdateExpiredTasksFunc != nil {
		return m.UpdateExpiredTasksFunc(ctx, threshold)
	}
	return 0, nil
}

func (m *TaskRepositoryMock) CountDatasetItems(ctx context.Context, datasetID int64) (int64, error) {
	if m.CountDatasetItemsFunc != nil {
		return m.CountDatasetItemsFunc(ctx, datasetID)
	}
	return 0, nil
}

func (m *TaskRepositoryMock) CountItemFiles(ctx context.Context, itemIDs []int64) (int64, error) {
	if m.CountItemFilesFunc != nil {
		return m.CountItemFilesFunc(ctx, itemIDs)
	}
	return 0, nil
}

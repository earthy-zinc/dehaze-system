package mock

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/repository"
)

// MockDatasetItemRepository 数据项仓储 Mock 实现
type MockDatasetItemRepository struct {
	FindByIDFunc          func(ctx context.Context, id int64) (*model.SysDatasetItem, error)
	FindByDatasetIDFunc   func(ctx context.Context, datasetID int64) ([]model.SysDatasetItem, error)
	CreateFunc            func(ctx context.Context, item *model.SysDatasetItem) error
	BatchCreateFunc       func(ctx context.Context, items []model.SysDatasetItem) error
	DeleteFunc            func(ctx context.Context, ids []int64) error
	DeleteByDatasetIDFunc func(ctx context.Context, datasetID int64) error
	UpdateFunc            func(ctx context.Context, item *model.SysDatasetItem) error
	FindPageFunc          func(ctx context.Context, datasetID int64, pageNum, pageSize int) ([]model.SysDatasetItem, int64, error)
}

func (m *MockDatasetItemRepository) FindByID(ctx context.Context, id int64) (*model.SysDatasetItem, error) {
	if m.FindByIDFunc != nil {
		return m.FindByIDFunc(ctx, id)
	}
	return nil, nil
}

func (m *MockDatasetItemRepository) FindByDatasetID(ctx context.Context, datasetID int64) ([]model.SysDatasetItem, error) {
	if m.FindByDatasetIDFunc != nil {
		return m.FindByDatasetIDFunc(ctx, datasetID)
	}
	return nil, nil
}

func (m *MockDatasetItemRepository) Create(ctx context.Context, item *model.SysDatasetItem) error {
	if m.CreateFunc != nil {
		return m.CreateFunc(ctx, item)
	}
	return nil
}

func (m *MockDatasetItemRepository) BatchCreate(ctx context.Context, items []model.SysDatasetItem) error {
	if m.BatchCreateFunc != nil {
		return m.BatchCreateFunc(ctx, items)
	}
	return nil
}

func (m *MockDatasetItemRepository) Delete(ctx context.Context, ids []int64) error {
	if m.DeleteFunc != nil {
		return m.DeleteFunc(ctx, ids)
	}
	return nil
}

func (m *MockDatasetItemRepository) DeleteByDatasetID(ctx context.Context, datasetID int64) error {
	if m.DeleteByDatasetIDFunc != nil {
		return m.DeleteByDatasetIDFunc(ctx, datasetID)
	}
	return nil
}

func (m *MockDatasetItemRepository) Update(ctx context.Context, item *model.SysDatasetItem) error {
	if m.UpdateFunc != nil {
		return m.UpdateFunc(ctx, item)
	}
	return nil
}

func (m *MockDatasetItemRepository) FindPage(ctx context.Context, datasetID int64, pageNum, pageSize int) ([]model.SysDatasetItem, int64, error) {
	if m.FindPageFunc != nil {
		return m.FindPageFunc(ctx, datasetID, pageNum, pageSize)
	}
	return nil, 0, nil
}

var _ repository.IDatasetItemRepository = (*MockDatasetItemRepository)(nil)

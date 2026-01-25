package mock

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
)

// MockDatasetRepository 数据集仓储 Mock 实现
type MockDatasetRepository struct {
	FindByIDFunc    func(ctx context.Context, id int64) (*model.SysDataset, error)
	FindPageFunc    func(ctx context.Context, q *query.DatasetQuery) (*vo.PageResult[vo.DatasetVO], error)
	CreateFunc      func(ctx context.Context, dataset *model.SysDataset) error
	UpdateFunc      func(ctx context.Context, dataset *model.SysDataset) error
	DeleteFunc      func(ctx context.Context, ids []int64) error
	GetFormDataFunc func(ctx context.Context, datasetID int64) (*bo.DatasetFormBO, error)
}

func (m *MockDatasetRepository) FindByID(ctx context.Context, id int64) (*model.SysDataset, error) {
	if m.FindByIDFunc != nil {
		return m.FindByIDFunc(ctx, id)
	}
	return nil, nil
}

func (m *MockDatasetRepository) FindPage(ctx context.Context, q *query.DatasetQuery) (*vo.PageResult[vo.DatasetVO], error) {
	if m.FindPageFunc != nil {
		return m.FindPageFunc(ctx, q)
	}
	return nil, nil
}

func (m *MockDatasetRepository) Create(ctx context.Context, dataset *model.SysDataset) error {
	if m.CreateFunc != nil {
		return m.CreateFunc(ctx, dataset)
	}
	return nil
}

func (m *MockDatasetRepository) Update(ctx context.Context, dataset *model.SysDataset) error {
	if m.UpdateFunc != nil {
		return m.UpdateFunc(ctx, dataset)
	}
	return nil
}

func (m *MockDatasetRepository) Delete(ctx context.Context, ids []int64) error {
	if m.DeleteFunc != nil {
		return m.DeleteFunc(ctx, ids)
	}
	return nil
}

func (m *MockDatasetRepository) GetFormData(ctx context.Context, datasetID int64) (*bo.DatasetFormBO, error) {
	if m.GetFormDataFunc != nil {
		return m.GetFormDataFunc(ctx, datasetID)
	}
	return nil, nil
}

var _ repository.IDatasetRepository = (*MockDatasetRepository)(nil)

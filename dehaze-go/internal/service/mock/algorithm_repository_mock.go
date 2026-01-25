package mock

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
)

// MockAlgorithmRepository 算法仓储 Mock 实现
type MockAlgorithmRepository struct {
	FindByIDFunc     func(ctx context.Context, id int64) (*model.SysAlgorithm, error)
	FindPageFunc     func(ctx context.Context, q *query.AlgorithmQuery) (*vo.PageResult[vo.AlgorithmVO], error)
	FindOptionsFunc  func(ctx context.Context) ([]vo.Option, error)
	CreateFunc       func(ctx context.Context, algorithm *model.SysAlgorithm) error
	UpdateFunc       func(ctx context.Context, algorithm *model.SysAlgorithm) error
	DeleteFunc       func(ctx context.Context, ids []int64) error
	UpdateStatusFunc func(ctx context.Context, id int64, status int8) error
}

func (m *MockAlgorithmRepository) FindByID(ctx context.Context, id int64) (*model.SysAlgorithm, error) {
	if m.FindByIDFunc != nil {
		return m.FindByIDFunc(ctx, id)
	}
	return nil, nil
}

func (m *MockAlgorithmRepository) FindPage(ctx context.Context, q *query.AlgorithmQuery) (*vo.PageResult[vo.AlgorithmVO], error) {
	if m.FindPageFunc != nil {
		return m.FindPageFunc(ctx, q)
	}
	return nil, nil
}

func (m *MockAlgorithmRepository) FindOptions(ctx context.Context) ([]vo.Option, error) {
	if m.FindOptionsFunc != nil {
		return m.FindOptionsFunc(ctx)
	}
	return nil, nil
}

func (m *MockAlgorithmRepository) Create(ctx context.Context, algorithm *model.SysAlgorithm) error {
	if m.CreateFunc != nil {
		return m.CreateFunc(ctx, algorithm)
	}
	return nil
}

func (m *MockAlgorithmRepository) Update(ctx context.Context, algorithm *model.SysAlgorithm) error {
	if m.UpdateFunc != nil {
		return m.UpdateFunc(ctx, algorithm)
	}
	return nil
}

func (m *MockAlgorithmRepository) Delete(ctx context.Context, ids []int64) error {
	if m.DeleteFunc != nil {
		return m.DeleteFunc(ctx, ids)
	}
	return nil
}

func (m *MockAlgorithmRepository) UpdateStatus(ctx context.Context, id int64, status int8) error {
	if m.UpdateStatusFunc != nil {
		return m.UpdateStatusFunc(ctx, id, status)
	}
	return nil
}

var _ repository.IAlgorithmRepository = (*MockAlgorithmRepository)(nil)

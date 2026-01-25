package mock

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
)

// MockDictRepository 字典数据仓储 Mock 实现
type MockDictRepository struct {
	FindByIDFunc       func(ctx context.Context, id int64) (*model.SysDict, error)
	FindByTypeCodeFunc func(ctx context.Context, typeCode string) ([]model.SysDict, error)
	FindPageFunc       func(ctx context.Context, q *query.DictPageQuery) (*vo.PageResult[vo.DictPageVO], error)
	CreateFunc         func(ctx context.Context, dict *model.SysDict) error
	UpdateFunc         func(ctx context.Context, dict *model.SysDict) error
	DeleteFunc         func(ctx context.Context, ids []int64) error
}

func (m *MockDictRepository) FindByID(ctx context.Context, id int64) (*model.SysDict, error) {
	if m.FindByIDFunc != nil {
		return m.FindByIDFunc(ctx, id)
	}
	return nil, nil
}

func (m *MockDictRepository) FindByTypeCode(ctx context.Context, typeCode string) ([]model.SysDict, error) {
	if m.FindByTypeCodeFunc != nil {
		return m.FindByTypeCodeFunc(ctx, typeCode)
	}
	return nil, nil
}

func (m *MockDictRepository) FindPage(ctx context.Context, q *query.DictPageQuery) (*vo.PageResult[vo.DictPageVO], error) {
	if m.FindPageFunc != nil {
		return m.FindPageFunc(ctx, q)
	}
	return nil, nil
}

func (m *MockDictRepository) Create(ctx context.Context, dict *model.SysDict) error {
	if m.CreateFunc != nil {
		return m.CreateFunc(ctx, dict)
	}
	return nil
}

func (m *MockDictRepository) Update(ctx context.Context, dict *model.SysDict) error {
	if m.UpdateFunc != nil {
		return m.UpdateFunc(ctx, dict)
	}
	return nil
}

func (m *MockDictRepository) Delete(ctx context.Context, ids []int64) error {
	if m.DeleteFunc != nil {
		return m.DeleteFunc(ctx, ids)
	}
	return nil
}

var _ repository.IDictRepository = (*MockDictRepository)(nil)

package mock

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
)

// MockDictTypeRepository 字典类型仓储 Mock 实现
type MockDictTypeRepository struct {
	FindByIDFunc     func(ctx context.Context, id int64) (*model.SysDictType, error)
	FindByCodeFunc   func(ctx context.Context, code string) (*model.SysDictType, error)
	ExistsByCodeFunc func(ctx context.Context, code string, excludeID ...int64) (bool, error)
	FindPageFunc     func(ctx context.Context, q *query.DictTypePageQuery) (*vo.PageResult[vo.DictTypePageVO], error)
	CreateFunc       func(ctx context.Context, dictType *model.SysDictType) error
	UpdateFunc       func(ctx context.Context, dictType *model.SysDictType) error
	DeleteFunc       func(ctx context.Context, ids []int64) error
}

func (m *MockDictTypeRepository) FindByID(ctx context.Context, id int64) (*model.SysDictType, error) {
	if m.FindByIDFunc != nil {
		return m.FindByIDFunc(ctx, id)
	}
	return nil, nil
}

func (m *MockDictTypeRepository) FindByCode(ctx context.Context, code string) (*model.SysDictType, error) {
	if m.FindByCodeFunc != nil {
		return m.FindByCodeFunc(ctx, code)
	}
	return nil, nil
}

func (m *MockDictTypeRepository) ExistsByCode(ctx context.Context, code string, excludeID ...int64) (bool, error) {
	if m.ExistsByCodeFunc != nil {
		return m.ExistsByCodeFunc(ctx, code, excludeID...)
	}
	return false, nil
}

func (m *MockDictTypeRepository) FindPage(ctx context.Context, q *query.DictTypePageQuery) (*vo.PageResult[vo.DictTypePageVO], error) {
	if m.FindPageFunc != nil {
		return m.FindPageFunc(ctx, q)
	}
	return nil, nil
}

func (m *MockDictTypeRepository) Create(ctx context.Context, dictType *model.SysDictType) error {
	if m.CreateFunc != nil {
		return m.CreateFunc(ctx, dictType)
	}
	return nil
}

func (m *MockDictTypeRepository) Update(ctx context.Context, dictType *model.SysDictType) error {
	if m.UpdateFunc != nil {
		return m.UpdateFunc(ctx, dictType)
	}
	return nil
}

func (m *MockDictTypeRepository) Delete(ctx context.Context, ids []int64) error {
	if m.DeleteFunc != nil {
		return m.DeleteFunc(ctx, ids)
	}
	return nil
}

var _ repository.IDictTypeRepository = (*MockDictTypeRepository)(nil)

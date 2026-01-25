package mock

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
)

// MockDeptRepository 部门仓储 Mock 实现
type MockDeptRepository struct {
	FindByIDFunc       func(ctx context.Context, id int64) (*model.SysDept, error)
	FindAllFunc        func(ctx context.Context, q *query.DeptQuery) ([]model.SysDept, error)
	FindByParentIDFunc func(ctx context.Context, parentID int64) ([]model.SysDept, error)
	CreateFunc         func(ctx context.Context, dept *model.SysDept) error
	UpdateFunc         func(ctx context.Context, dept *model.SysDept) error
	DeleteFunc         func(ctx context.Context, id int64) error
	HasChildrenFunc    func(ctx context.Context, id int64) (bool, error)
	HasUsersFunc       func(ctx context.Context, deptID int64) (bool, error)
	GetOptionsFunc     func(ctx context.Context) ([]vo.Option, error)
	GetFormDataFunc    func(ctx context.Context, deptID int64) (*bo.DeptFormBO, error)
	GetSubDeptIDsFunc  func(ctx context.Context, deptID int64) ([]int64, error)
}

func (m *MockDeptRepository) FindByID(ctx context.Context, id int64) (*model.SysDept, error) {
	if m.FindByIDFunc != nil {
		return m.FindByIDFunc(ctx, id)
	}
	return nil, nil
}

func (m *MockDeptRepository) FindAll(ctx context.Context, q *query.DeptQuery) ([]model.SysDept, error) {
	if m.FindAllFunc != nil {
		return m.FindAllFunc(ctx, q)
	}
	return nil, nil
}

func (m *MockDeptRepository) FindByParentID(ctx context.Context, parentID int64) ([]model.SysDept, error) {
	if m.FindByParentIDFunc != nil {
		return m.FindByParentIDFunc(ctx, parentID)
	}
	return nil, nil
}

func (m *MockDeptRepository) Create(ctx context.Context, dept *model.SysDept) error {
	if m.CreateFunc != nil {
		return m.CreateFunc(ctx, dept)
	}
	return nil
}

func (m *MockDeptRepository) Update(ctx context.Context, dept *model.SysDept) error {
	if m.UpdateFunc != nil {
		return m.UpdateFunc(ctx, dept)
	}
	return nil
}

func (m *MockDeptRepository) Delete(ctx context.Context, id int64) error {
	if m.DeleteFunc != nil {
		return m.DeleteFunc(ctx, id)
	}
	return nil
}

func (m *MockDeptRepository) HasChildren(ctx context.Context, id int64) (bool, error) {
	if m.HasChildrenFunc != nil {
		return m.HasChildrenFunc(ctx, id)
	}
	return false, nil
}

func (m *MockDeptRepository) HasUsers(ctx context.Context, deptID int64) (bool, error) {
	if m.HasUsersFunc != nil {
		return m.HasUsersFunc(ctx, deptID)
	}
	return false, nil
}

func (m *MockDeptRepository) GetOptions(ctx context.Context) ([]vo.Option, error) {
	if m.GetOptionsFunc != nil {
		return m.GetOptionsFunc(ctx)
	}
	return nil, nil
}

func (m *MockDeptRepository) GetFormData(ctx context.Context, deptID int64) (*bo.DeptFormBO, error) {
	if m.GetFormDataFunc != nil {
		return m.GetFormDataFunc(ctx, deptID)
	}
	return nil, nil
}

func (m *MockDeptRepository) GetSubDeptIDs(ctx context.Context, deptID int64) ([]int64, error) {
	if m.GetSubDeptIDsFunc != nil {
		return m.GetSubDeptIDsFunc(ctx, deptID)
	}
	return nil, nil
}

var _ repository.IDeptRepository = (*MockDeptRepository)(nil)

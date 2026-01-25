package mock

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
)

// MockMenuRepository 菜单仓储 Mock 实现
type MockMenuRepository struct {
	FindByIDFunc          func(ctx context.Context, id int64) (*model.SysMenu, error)
	FindAllFunc           func(ctx context.Context, q *query.MenuQuery) ([]model.SysMenu, error)
	FindByParentIDFunc    func(ctx context.Context, parentID int64) ([]model.SysMenu, error)
	CreateFunc            func(ctx context.Context, menu *model.SysMenu) error
	UpdateFunc            func(ctx context.Context, menu *model.SysMenu) error
	DeleteFunc            func(ctx context.Context, id int64) error
	HasChildrenFunc       func(ctx context.Context, id int64) (bool, error)
	FindRoutesByRolesFunc func(ctx context.Context, roles []string) ([]model.SysMenu, error)
	FindPermsByRolesFunc  func(ctx context.Context, roles []string) ([]string, error)
	GetOptionsFunc        func(ctx context.Context) ([]vo.Option, error)
	GetFormDataFunc       func(ctx context.Context, menuID int64) (*bo.MenuForm, error)
}

func (m *MockMenuRepository) FindByID(ctx context.Context, id int64) (*model.SysMenu, error) {
	if m.FindByIDFunc != nil {
		return m.FindByIDFunc(ctx, id)
	}
	return nil, nil
}

func (m *MockMenuRepository) FindAll(ctx context.Context, q *query.MenuQuery) ([]model.SysMenu, error) {
	if m.FindAllFunc != nil {
		return m.FindAllFunc(ctx, q)
	}
	return nil, nil
}

func (m *MockMenuRepository) FindByParentID(ctx context.Context, parentID int64) ([]model.SysMenu, error) {
	if m.FindByParentIDFunc != nil {
		return m.FindByParentIDFunc(ctx, parentID)
	}
	return nil, nil
}

func (m *MockMenuRepository) Create(ctx context.Context, menu *model.SysMenu) error {
	if m.CreateFunc != nil {
		return m.CreateFunc(ctx, menu)
	}
	return nil
}

func (m *MockMenuRepository) Update(ctx context.Context, menu *model.SysMenu) error {
	if m.UpdateFunc != nil {
		return m.UpdateFunc(ctx, menu)
	}
	return nil
}

func (m *MockMenuRepository) Delete(ctx context.Context, id int64) error {
	if m.DeleteFunc != nil {
		return m.DeleteFunc(ctx, id)
	}
	return nil
}

func (m *MockMenuRepository) HasChildren(ctx context.Context, id int64) (bool, error) {
	if m.HasChildrenFunc != nil {
		return m.HasChildrenFunc(ctx, id)
	}
	return false, nil
}

func (m *MockMenuRepository) FindRoutesByRoles(ctx context.Context, roles []string) ([]model.SysMenu, error) {
	if m.FindRoutesByRolesFunc != nil {
		return m.FindRoutesByRolesFunc(ctx, roles)
	}
	return nil, nil
}

func (m *MockMenuRepository) FindPermsByRoles(ctx context.Context, roles []string) ([]string, error) {
	if m.FindPermsByRolesFunc != nil {
		return m.FindPermsByRolesFunc(ctx, roles)
	}
	return nil, nil
}

func (m *MockMenuRepository) GetOptions(ctx context.Context) ([]vo.Option, error) {
	if m.GetOptionsFunc != nil {
		return m.GetOptionsFunc(ctx)
	}
	return nil, nil
}

func (m *MockMenuRepository) GetFormData(ctx context.Context, menuID int64) (*bo.MenuForm, error) {
	if m.GetFormDataFunc != nil {
		return m.GetFormDataFunc(ctx, menuID)
	}
	return nil, nil
}

var _ repository.IMenuRepository = (*MockMenuRepository)(nil)

package mock

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
)

// MockRoleRepository 角色仓储 Mock 实现
type MockRoleRepository struct {
	FindByIDFunc     func(ctx context.Context, id int64) (*model.SysRole, error)
	FindByCodeFunc   func(ctx context.Context, code string) (*model.SysRole, error)
	ExistsByCodeFunc func(ctx context.Context, code string, excludeID ...int64) (bool, error)
	ExistsByNameFunc func(ctx context.Context, name string, excludeID ...int64) (bool, error)
	FindPageFunc     func(ctx context.Context, q *query.RolePageQuery) (*vo.PageResult[vo.RolePageVO], error)
	FindOptionsFunc  func(ctx context.Context) ([]vo.Option, error)
	CreateFunc       func(ctx context.Context, role *model.SysRole) error
	UpdateFunc       func(ctx context.Context, role *model.SysRole) error
	UpdateStatusFunc func(ctx context.Context, id int64, status int8) error
	DeleteFunc       func(ctx context.Context, ids []int64) error
	HasUsersFunc     func(ctx context.Context, roleID int64) (bool, error)
	GetMenuIDsFunc   func(ctx context.Context, roleID int64) ([]int64, error)
	AssignMenusFunc  func(ctx context.Context, roleID int64, menuIDs []int64) error
	GetFormDataFunc  func(ctx context.Context, roleID int64) (*bo.RoleFormBO, error)
}

func (m *MockRoleRepository) FindByID(ctx context.Context, id int64) (*model.SysRole, error) {
	if m.FindByIDFunc != nil {
		return m.FindByIDFunc(ctx, id)
	}
	return nil, nil
}

func (m *MockRoleRepository) FindByCode(ctx context.Context, code string) (*model.SysRole, error) {
	if m.FindByCodeFunc != nil {
		return m.FindByCodeFunc(ctx, code)
	}
	return nil, nil
}

func (m *MockRoleRepository) ExistsByCode(ctx context.Context, code string, excludeID ...int64) (bool, error) {
	if m.ExistsByCodeFunc != nil {
		return m.ExistsByCodeFunc(ctx, code, excludeID...)
	}
	return false, nil
}

func (m *MockRoleRepository) ExistsByName(ctx context.Context, name string, excludeID ...int64) (bool, error) {
	if m.ExistsByNameFunc != nil {
		return m.ExistsByNameFunc(ctx, name, excludeID...)
	}
	return false, nil
}

func (m *MockRoleRepository) FindPage(ctx context.Context, q *query.RolePageQuery) (*vo.PageResult[vo.RolePageVO], error) {
	if m.FindPageFunc != nil {
		return m.FindPageFunc(ctx, q)
	}
	return nil, nil
}

func (m *MockRoleRepository) FindOptions(ctx context.Context) ([]vo.Option, error) {
	if m.FindOptionsFunc != nil {
		return m.FindOptionsFunc(ctx)
	}
	return nil, nil
}

func (m *MockRoleRepository) Create(ctx context.Context, role *model.SysRole) error {
	if m.CreateFunc != nil {
		return m.CreateFunc(ctx, role)
	}
	return nil
}

func (m *MockRoleRepository) Update(ctx context.Context, role *model.SysRole) error {
	if m.UpdateFunc != nil {
		return m.UpdateFunc(ctx, role)
	}
	return nil
}

func (m *MockRoleRepository) UpdateStatus(ctx context.Context, id int64, status int8) error {
	if m.UpdateStatusFunc != nil {
		return m.UpdateStatusFunc(ctx, id, status)
	}
	return nil
}

func (m *MockRoleRepository) Delete(ctx context.Context, ids []int64) error {
	if m.DeleteFunc != nil {
		return m.DeleteFunc(ctx, ids)
	}
	return nil
}

func (m *MockRoleRepository) HasUsers(ctx context.Context, roleID int64) (bool, error) {
	if m.HasUsersFunc != nil {
		return m.HasUsersFunc(ctx, roleID)
	}
	return false, nil
}

func (m *MockRoleRepository) GetMenuIDs(ctx context.Context, roleID int64) ([]int64, error) {
	if m.GetMenuIDsFunc != nil {
		return m.GetMenuIDsFunc(ctx, roleID)
	}
	return nil, nil
}

func (m *MockRoleRepository) AssignMenus(ctx context.Context, roleID int64, menuIDs []int64) error {
	if m.AssignMenusFunc != nil {
		return m.AssignMenusFunc(ctx, roleID, menuIDs)
	}
	return nil
}

func (m *MockRoleRepository) GetFormData(ctx context.Context, roleID int64) (*bo.RoleFormBO, error) {
	if m.GetFormDataFunc != nil {
		return m.GetFormDataFunc(ctx, roleID)
	}
	return nil, nil
}

var _ repository.IRoleRepository = (*MockRoleRepository)(nil)

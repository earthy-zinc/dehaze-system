package mock

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
)

// MockUserRepository 用户仓储 Mock 实现
type MockUserRepository struct {
	FindByIDFunc         func(ctx context.Context, id int64) (*model.SysUser, error)
	FindByUsernameFunc   func(ctx context.Context, username string) (*model.SysUser, error)
	ExistsByUsernameFunc func(ctx context.Context, username string, excludeID ...int64) (bool, error)
	FindPageFunc         func(ctx context.Context, q *query.UserPageQuery) (*vo.PageResult[vo.UserPageVO], error)
	CreateFunc           func(ctx context.Context, user *model.SysUser) error
	UpdateFunc           func(ctx context.Context, user *model.SysUser) error
	UpdateStatusFunc     func(ctx context.Context, id int64, status int8) error
	UpdatePasswordFunc   func(ctx context.Context, id int64, password string) error
	DeleteFunc           func(ctx context.Context, ids []int64) error
	FindUserAuthInfoFunc func(ctx context.Context, username string) (*model.UserAuthInfo, error)
	AssignRolesFunc      func(ctx context.Context, userID int64, roleIDs []int64) error
	GetUserRoleIDsFunc   func(ctx context.Context, userID int64) ([]int64, error)
	GetFormDataFunc      func(ctx context.Context, userID int64) (*bo.UserFormBO, error)
}

func (m *MockUserRepository) FindByID(ctx context.Context, id int64) (*model.SysUser, error) {
	if m.FindByIDFunc != nil {
		return m.FindByIDFunc(ctx, id)
	}
	return nil, nil
}

func (m *MockUserRepository) FindByUsername(ctx context.Context, username string) (*model.SysUser, error) {
	if m.FindByUsernameFunc != nil {
		return m.FindByUsernameFunc(ctx, username)
	}
	return nil, nil
}

func (m *MockUserRepository) ExistsByUsername(ctx context.Context, username string, excludeID ...int64) (bool, error) {
	if m.ExistsByUsernameFunc != nil {
		return m.ExistsByUsernameFunc(ctx, username, excludeID...)
	}
	return false, nil
}

func (m *MockUserRepository) FindPage(ctx context.Context, q *query.UserPageQuery) (*vo.PageResult[vo.UserPageVO], error) {
	if m.FindPageFunc != nil {
		return m.FindPageFunc(ctx, q)
	}
	return nil, nil
}

func (m *MockUserRepository) Create(ctx context.Context, user *model.SysUser) error {
	if m.CreateFunc != nil {
		return m.CreateFunc(ctx, user)
	}
	return nil
}

func (m *MockUserRepository) Update(ctx context.Context, user *model.SysUser) error {
	if m.UpdateFunc != nil {
		return m.UpdateFunc(ctx, user)
	}
	return nil
}

func (m *MockUserRepository) UpdateStatus(ctx context.Context, id int64, status int8) error {
	if m.UpdateStatusFunc != nil {
		return m.UpdateStatusFunc(ctx, id, status)
	}
	return nil
}

func (m *MockUserRepository) UpdatePassword(ctx context.Context, id int64, password string) error {
	if m.UpdatePasswordFunc != nil {
		return m.UpdatePasswordFunc(ctx, id, password)
	}
	return nil
}

func (m *MockUserRepository) Delete(ctx context.Context, ids []int64) error {
	if m.DeleteFunc != nil {
		return m.DeleteFunc(ctx, ids)
	}
	return nil
}

func (m *MockUserRepository) FindUserAuthInfo(ctx context.Context, username string) (*model.UserAuthInfo, error) {
	if m.FindUserAuthInfoFunc != nil {
		return m.FindUserAuthInfoFunc(ctx, username)
	}
	return nil, nil
}

func (m *MockUserRepository) AssignRoles(ctx context.Context, userID int64, roleIDs []int64) error {
	if m.AssignRolesFunc != nil {
		return m.AssignRolesFunc(ctx, userID, roleIDs)
	}
	return nil
}

func (m *MockUserRepository) GetUserRoleIDs(ctx context.Context, userID int64) ([]int64, error) {
	if m.GetUserRoleIDsFunc != nil {
		return m.GetUserRoleIDsFunc(ctx, userID)
	}
	return nil, nil
}

func (m *MockUserRepository) GetFormData(ctx context.Context, userID int64) (*bo.UserFormBO, error) {
	if m.GetFormDataFunc != nil {
		return m.GetFormDataFunc(ctx, userID)
	}
	return nil, nil
}

var _ repository.IUserRepository = (*MockUserRepository)(nil)

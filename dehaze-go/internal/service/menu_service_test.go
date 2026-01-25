package service

import (
	"context"
	"errors"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/repository"
	mock_repository "github.com/earthyzinc/dehaze-go/internal/service/mock"
	"github.com/stretchr/testify/assert"
	"gorm.io/gorm"
)

func TestGetList_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockMenus := []model.SysMenu{
		{BaseModel: model.BaseModel{ID: 1}, ParentID: 0, Name: "系统管理", Type: 2, Path: "/system", Sort: 1},
		{BaseModel: model.BaseModel{ID: 2}, ParentID: 1, Name: "用户管理", Type: 1, Path: "/user", Component: "system/user/index", Sort: 1},
		{BaseModel: model.BaseModel{ID: 3}, ParentID: 1, Name: "角色管理", Type: 1, Path: "/role", Component: "system/role/index", Sort: 2},
	}

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.MenuQuery) ([]model.SysMenu, error) {
		return mockMenus, nil
	}

	result, err := menuService.GetList(ctx, &query.MenuQuery{})

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result, 1)
	assert.Equal(t, "系统管理", result[0].Name)
	assert.Len(t, result[0].Children, 2)
	assert.Equal(t, "用户管理", result[0].Children[0].Name)
	assert.Equal(t, "角色管理", result[0].Children[1].Name)
}

func TestGetList_WithKeywords(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockMenus := []model.SysMenu{
		{BaseModel: model.BaseModel{ID: 1}, ParentID: 0, Name: "系统管理", Type: 2, Path: "/system", Sort: 1},
		{BaseModel: model.BaseModel{ID: 2}, ParentID: 1, Name: "用户管理", Type: 1, Path: "/user", Component: "system/user/index", Sort: 1},
	}

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.MenuQuery) ([]model.SysMenu, error) {
		assert.Equal(t, "用户", q.Keywords)
		return mockMenus, nil
	}

	result, err := menuService.GetList(ctx, &query.MenuQuery{Keywords: "用户"})

	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestGetList_Empty(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.MenuQuery) ([]model.SysMenu, error) {
		return []model.SysMenu{}, nil
	}

	result, err := menuService.GetList(ctx, &query.MenuQuery{})

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Empty(t, result)
}

func TestMenuGetFormData_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	expectedForm := &bo.MenuForm{
		ID:       pointerToInt64(1),
		ParentID: 0,
		Name:     "系统管理",
		Type:     2,
		Path:     "/system",
		Visible:  1,
		Sort:     1,
	}

	mockRepo.GetFormDataFunc = func(ctx context.Context, menuID int64) (*bo.MenuForm, error) {
		assert.Equal(t, int64(1), menuID)
		return expectedForm, nil
	}

	result, err := menuService.GetFormData(ctx, 1)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, expectedForm.Name, result.Name)
	assert.Equal(t, expectedForm.Type, result.Type)
}

func TestMenuGetFormData_NotFound(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockRepo.GetFormDataFunc = func(ctx context.Context, menuID int64) (*bo.MenuForm, error) {
		return nil, nil
	}

	result, err := menuService.GetFormData(ctx, 999)

	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "菜单不存在")
}

func TestMenuCreate_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysMenu, error) {
		if id == 0 {
			return nil, nil
		}
		return &model.SysMenu{BaseModel: model.BaseModel{ID: 1}, TreePath: "0"}, nil
	}

	mockRepo.CreateFunc = func(ctx context.Context, menu *model.SysMenu) error {
		assert.Equal(t, "测试菜单", menu.Name)
		assert.Equal(t, int8(1), menu.Type)
		assert.Equal(t, "/test", menu.Path)
		return nil
	}

	form := &bo.MenuForm{
		ParentID: 1,
		Name:     "测试菜单",
		Type:     1,
		Path:     "/test",
		Visible:  1,
		Sort:     1,
	}

	err := menuService.Create(ctx, form)

	assert.NoError(t, err)
}

func TestMenuCreate_Directory(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockRepo.CreateFunc = func(ctx context.Context, menu *model.SysMenu) error {
		assert.Equal(t, "系统管理", menu.Name)
		assert.Equal(t, int8(2), menu.Type)
		assert.Equal(t, "/system", menu.Path)
		assert.Equal(t, "Layout", menu.Component)
		return nil
	}

	form := &bo.MenuForm{
		ParentID:  0,
		Name:      "系统管理",
		Type:      2,
		Path:      "/system",
		Component: "CustomComponent",
		Visible:   1,
		Sort:      1,
	}

	err := menuService.Create(ctx, form)

	assert.NoError(t, err)
}

func TestMenuCreate_Button(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockRepo.CreateFunc = func(ctx context.Context, menu *model.SysMenu) error {
		assert.Equal(t, "添加用户", menu.Name)
		assert.Equal(t, int8(4), menu.Type)
		assert.Empty(t, menu.Component)
		return nil
	}

	form := &bo.MenuForm{
		ParentID: 1,
		Name:     "添加用户",
		Type:     4,
		Perm:     "user:add",
		Visible:  1,
		Sort:     1,
	}

	err := menuService.Create(ctx, form)

	assert.NoError(t, err)
}

func TestMenuCreate_NilForm(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	err := menuService.Create(ctx, nil)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "表单数据不能为空")
}

func TestMenuUpdate_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysMenu, error) {
		if id == 0 {
			return nil, nil
		}
		return &model.SysMenu{BaseModel: model.BaseModel{ID: 1}, TreePath: "0"}, nil
	}

	mockRepo.UpdateFunc = func(ctx context.Context, menu *model.SysMenu) error {
		assert.Equal(t, int64(1), menu.ID)
		assert.Equal(t, "更新后的菜单", menu.Name)
		return nil
	}

	form := &bo.MenuForm{
		ParentID: 0,
		Name:     "更新后的菜单",
		Type:     1,
		Path:     "/updated",
		Visible:  1,
		Sort:     1,
	}

	err := menuService.Update(ctx, 1, form)

	assert.NoError(t, err)
}

func TestMenuUpdate_NilForm(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	err := menuService.Update(ctx, 1, nil)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "表单数据不能为空")
}

func TestMenuDelete_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockRepo.DeleteFunc = func(ctx context.Context, id int64) error {
		assert.Equal(t, int64(1), id)
		return nil
	}

	err := menuService.Delete(ctx, 1)

	assert.NoError(t, err)
}

func TestMenuDelete_RepositoryError(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockRepo.DeleteFunc = func(ctx context.Context, id int64) error {
		return errors.New("database error")
	}

	err := menuService.Delete(ctx, 1)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "database error")
}

func TestMenuGetOptions_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockMenus := []model.SysMenu{
		{BaseModel: model.BaseModel{ID: 1}, ParentID: 0, Name: "系统管理", Sort: 1},
		{BaseModel: model.BaseModel{ID: 2}, ParentID: 1, Name: "用户管理", Sort: 1},
		{BaseModel: model.BaseModel{ID: 3}, ParentID: 1, Name: "角色管理", Sort: 2},
	}

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.MenuQuery) ([]model.SysMenu, error) {
		return mockMenus, nil
	}

	options, err := menuService.GetOptions(ctx)

	assert.NoError(t, err)
	assert.NotNil(t, options)
	assert.Len(t, options, 1)
	assert.Equal(t, "系统管理", options[0].Label)
	assert.Len(t, options[0].Children, 2)
	assert.Equal(t, "用户管理", options[0].Children[0].Label)
}

func TestGetRoutes_Success(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockMenus := []model.SysMenu{
		{BaseModel: model.BaseModel{ID: 1}, ParentID: 0, Name: "系统管理", Type: 2, Path: "/system", Sort: 1},
		{BaseModel: model.BaseModel{ID: 2}, ParentID: 1, Name: "用户管理", Type: 1, Path: "/user", Component: "system/user/index", Sort: 1, KeepAlive: 1},
	}

	mockRepo.FindRoutesByRolesFunc = func(ctx context.Context, roles []string) ([]model.SysMenu, error) {
		assert.Contains(t, roles, "ADMIN")
		return mockMenus, nil
	}

	routes, err := menuService.GetRoutes(ctx, []string{"ADMIN"})

	assert.NoError(t, err)
	assert.NotNil(t, routes)
	assert.Len(t, routes, 1)
	assert.Equal(t, "/system", routes[0].Path)
	assert.Len(t, routes[0].Children, 1)
	assert.Equal(t, "/user", routes[0].Children[0].Path)
	assert.NotNil(t, routes[0].Children[0].Meta.KeepAlive)
	assert.True(t, *routes[0].Children[0].Meta.KeepAlive)
}

func TestGetRoutes_EmptyRoles(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockMenus := []model.SysMenu{}

	mockRepo.FindRoutesByRolesFunc = func(ctx context.Context, roles []string) ([]model.SysMenu, error) {
		assert.Empty(t, roles)
		return mockMenus, nil
	}

	routes, err := menuService.GetRoutes(ctx, []string{})

	assert.NoError(t, err)
	assert.NotNil(t, routes)
	assert.Empty(t, routes)
}

func TestGetRoutes_RepositoryError(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockRepo.FindRoutesByRolesFunc = func(ctx context.Context, roles []string) ([]model.SysMenu, error) {
		return nil, errors.New("database error")
	}

	routes, err := menuService.GetRoutes(ctx, []string{"ADMIN"})

	assert.Error(t, err)
	assert.Nil(t, routes)
	assert.Contains(t, err.Error(), "database error")
}

func TestGetFormData_GormError(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockRepo.GetFormDataFunc = func(ctx context.Context, menuID int64) (*bo.MenuForm, error) {
		return nil, gorm.ErrRecordNotFound
	}

	result, err := menuService.GetFormData(ctx, 999)

	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "菜单不存在")
}

func TestMenuGetOptions_Empty(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.MenuQuery) ([]model.SysMenu, error) {
		return []model.SysMenu{}, nil
	}

	options, err := menuService.GetOptions(ctx)

	assert.NoError(t, err)
	assert.NotNil(t, options)
	assert.Empty(t, options)
}

func TestGetList_RepositoryError(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.MenuQuery) ([]model.SysMenu, error) {
		return nil, errors.New("database error")
	}

	result, err := menuService.GetList(ctx, &query.MenuQuery{})

	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "database error")
}

func TestCreate_RepositoryError(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysMenu, error) {
		return nil, nil
	}

	mockRepo.CreateFunc = func(ctx context.Context, menu *model.SysMenu) error {
		return errors.New("database error")
	}

	form := &bo.MenuForm{
		ParentID: 0,
		Name:     "测试菜单",
		Type:     1,
		Path:     "/test",
		Visible:  1,
		Sort:     1,
	}

	err := menuService.Create(ctx, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "database error")
}

func TestUpdate_RepositoryError(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysMenu, error) {
		return nil, nil
	}

	mockRepo.UpdateFunc = func(ctx context.Context, menu *model.SysMenu) error {
		return errors.New("database error")
	}

	form := &bo.MenuForm{
		ParentID: 0,
		Name:     "更新后的菜单",
		Type:     1,
		Path:     "/updated",
		Visible:  1,
		Sort:     1,
	}

	err := menuService.Update(ctx, 1, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "database error")
}

func TestMenuGetOptions_RepositoryError(t *testing.T) {
	ctx := context.Background()
	mockRepo := new(mock_repository.MockMenuRepository)
	menuService := NewMenuService(mockRepo)

	mockRepo.FindAllFunc = func(ctx context.Context, q *query.MenuQuery) ([]model.SysMenu, error) {
		return nil, errors.New("database error")
	}

	options, err := menuService.GetOptions(ctx)

	assert.Error(t, err)
	assert.Nil(t, options)
	assert.Contains(t, err.Error(), "database error")
}

func pointerToInt64(n int64) *int64 {
	return &n
}

var _ repository.IMenuRepository = (*mock_repository.MockMenuRepository)(nil)

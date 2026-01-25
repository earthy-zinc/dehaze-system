package service

import (
	"context"
	"errors"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	mock_repository "github.com/earthyzinc/dehaze-go/internal/service/mock"
	"github.com/stretchr/testify/assert"
)

func TestRoleGetPage_Success(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	expectedResult := &vo.PageResult[vo.RolePageVO]{
		List: []vo.RolePageVO{
			{ID: 1, Name: "管理员", Code: "ADMIN", DataScope: 0},
			{ID: 2, Name: "普通用户", Code: "USER", DataScope: 3},
		},
		Total: 2,
	}

	mockRoleRepo.FindPageFunc = func(ctx context.Context, q *query.RolePageQuery) (*vo.PageResult[vo.RolePageVO], error) {
		return expectedResult, nil
	}

	result, err := roleService.GetPage(ctx, &query.RolePageQuery{})

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Len(t, result.List, 2)
	assert.Equal(t, "全部数据", result.List[0].DataScopeLabel)
	assert.Equal(t, "本人数据", result.List[1].DataScopeLabel)
}

func TestRoleCreate_Success(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	form := &bo.RoleFormBO{
		Name:      "测试角色",
		Code:      "TEST_ROLE",
		Sort:      1,
		Status:    1,
		DataScope: 2,
	}

	mockRoleRepo.ExistsByCodeFunc = func(ctx context.Context, code string, excludeID ...int64) (bool, error) {
		return false, nil
	}

	mockRoleRepo.ExistsByNameFunc = func(ctx context.Context, name string, excludeID ...int64) (bool, error) {
		return false, nil
	}

	mockRoleRepo.CreateFunc = func(ctx context.Context, role *model.SysRole) error {
		assert.Equal(t, form.Name, role.Name)
		assert.Equal(t, form.Code, role.Code)
		assert.Equal(t, form.Sort, role.Sort)
		assert.Equal(t, form.Status, role.Status)
		assert.Equal(t, form.DataScope, role.DataScope)
		return nil
	}

	err := roleService.Create(ctx, form)

	assert.NoError(t, err)
}

func TestRoleCreate_DuplicateCode(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	form := &bo.RoleFormBO{
		Name:   "测试角色",
		Code:   "ADMIN",
		Sort:   1,
		Status: 1,
	}

	mockRoleRepo.ExistsByCodeFunc = func(ctx context.Context, code string, excludeID ...int64) (bool, error) {
		return true, nil
	}

	err := roleService.Create(ctx, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "角色编码已存在")
}

func TestRoleCreate_DuplicateName(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	form := &bo.RoleFormBO{
		Name:   "管理员",
		Code:   "TEST_ROLE",
		Sort:   1,
		Status: 1,
	}

	mockRoleRepo.ExistsByCodeFunc = func(ctx context.Context, code string, excludeID ...int64) (bool, error) {
		return false, nil
	}

	mockRoleRepo.ExistsByNameFunc = func(ctx context.Context, name string, excludeID ...int64) (bool, error) {
		return true, nil
	}

	err := roleService.Create(ctx, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "角色名称已存在")
}

func TestRoleCreate_InvalidForm(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	tests := []struct {
		name        string
		form        *bo.RoleFormBO
		expectedErr string
	}{
		{
			name:        "空名称",
			form:        &bo.RoleFormBO{Name: "", Code: "TEST"},
			expectedErr: "角色名称不能为空",
		},
		{
			name:        "名称太短",
			form:        &bo.RoleFormBO{Name: "A", Code: "TEST"},
			expectedErr: "角色名称长度必须在2-30个字符之间",
		},
		{
			name:        "空编码",
			form:        &bo.RoleFormBO{Name: "测试", Code: ""},
			expectedErr: "角色编码不能为空",
		},
		{
			name:        "编码格式错误",
			form:        &bo.RoleFormBO{Name: "测试", Code: "test_role"},
			expectedErr: "角色编码格式不正确",
		},
		{
			name:        "数据权限范围无效",
			form:        &bo.RoleFormBO{Name: "测试", Code: "TEST", DataScope: 5},
			expectedErr: "数据权限范围值无效",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := roleService.Create(ctx, tt.form)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectedErr)
		})
	}
}

func TestRoleUpdate_Success(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	form := &bo.RoleFormBO{
		Name:      "更新后的角色",
		Code:      "ADMIN",
		Sort:      1,
		Status:    1,
		DataScope: 2,
	}

	mockRoleRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysRole, error) {
		return &model.SysRole{
			BaseModel: model.BaseModel{ID: id},
			Name:      "旧角色名",
			Code:      "ADMIN",
			Sort:      1,
			Status:    1,
			DataScope: 1,
		}, nil
	}

	mockRoleRepo.ExistsByCodeFunc = func(ctx context.Context, code string, excludeID ...int64) (bool, error) {
		return false, nil
	}

	mockRoleRepo.ExistsByNameFunc = func(ctx context.Context, name string, excludeID ...int64) (bool, error) {
		return false, nil
	}

	mockRoleRepo.UpdateFunc = func(ctx context.Context, role *model.SysRole) error {
		assert.Equal(t, int64(1), role.ID)
		assert.Equal(t, form.Name, role.Name)
		assert.Equal(t, form.Code, role.Code)
		return nil
	}

	err := roleService.Update(ctx, 1, form)

	assert.NoError(t, err)
}

func TestRoleUpdate_RoleNotFound(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	form := &bo.RoleFormBO{Name: "测试", Code: "TEST"}

	mockRoleRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysRole, error) {
		return nil, nil
	}

	err := roleService.Update(ctx, 999, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "角色不存在")
}

func TestRoleUpdate_RootRoleProtected(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	form := &bo.RoleFormBO{Name: "ROOT", Code: "ROOT"}

	mockRoleRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysRole, error) {
		return &model.SysRole{
			BaseModel: model.BaseModel{ID: 1},
			Code:      "ROOT",
			Name:      "超级管理员",
		}, nil
	}

	err := roleService.Update(ctx, 1, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "超级管理员角色不可修改")
}

func TestRoleUpdate_CodeChanged(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	form := &bo.RoleFormBO{Name: "测试", Code: "NEW_CODE"}

	mockRoleRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysRole, error) {
		return &model.SysRole{
			BaseModel: model.BaseModel{ID: 1},
			Code:      "OLD_CODE",
			Name:      "测试",
		}, nil
	}

	err := roleService.Update(ctx, 1, form)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "角色编码不可修改")
}

func TestRoleDelete_Success(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	mockRoleRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysRole, error) {
		return &model.SysRole{
			BaseModel: model.BaseModel{ID: id},
			Code:      "TEST_ROLE",
			Name:      "测试角色",
		}, nil
	}

	mockRoleRepo.HasUsersFunc = func(ctx context.Context, roleID int64) (bool, error) {
		return false, nil
	}

	mockRoleRepo.DeleteFunc = func(ctx context.Context, ids []int64) error {
		assert.Equal(t, []int64{1, 2}, ids)
		return nil
	}

	err := roleService.Delete(ctx, []int64{1, 2})

	assert.NoError(t, err)
}

func TestRoleDelete_RootRoleProtected(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	mockRoleRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysRole, error) {
		return &model.SysRole{
			BaseModel: model.BaseModel{ID: id},
			Code:      "ROOT",
			Name:      "超级管理员",
		}, nil
	}

	err := roleService.Delete(ctx, []int64{1})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "超级管理员角色不可删除")
}

func TestRoleDelete_HasUsers(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	mockRoleRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysRole, error) {
		return &model.SysRole{
			BaseModel: model.BaseModel{ID: id},
			Code:      "TEST_ROLE",
			Name:      "测试角色",
		}, nil
	}

	mockRoleRepo.HasUsersFunc = func(ctx context.Context, roleID int64) (bool, error) {
		return true, nil
	}

	err := roleService.Delete(ctx, []int64{1})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "已分配用户")
}

func TestRoleDelete_EmptyIDs(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	err := roleService.Delete(ctx, []int64{})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "删除的角色ID不能为空")
}

func TestRoleUpdateStatus_Success(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	mockRoleRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysRole, error) {
		return &model.SysRole{
			BaseModel: model.BaseModel{ID: id},
			Code:      "TEST_ROLE",
			Name:      "测试角色",
			Status:    1,
		}, nil
	}

	mockRoleRepo.UpdateStatusFunc = func(ctx context.Context, id int64, status int8) error {
		assert.Equal(t, int64(1), id)
		assert.Equal(t, int8(0), status)
		return nil
	}

	err := roleService.UpdateStatus(ctx, 1, 0)

	assert.NoError(t, err)
}

func TestRoleUpdateStatus_RootRoleProtected(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	mockRoleRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysRole, error) {
		return &model.SysRole{
			BaseModel: model.BaseModel{ID: id},
			Code:      "ROOT",
			Name:      "超级管理员",
			Status:    1,
		}, nil
	}

	err := roleService.UpdateStatus(ctx, 1, 0)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "超级管理员角色不可修改状态")
}

func TestRoleUpdateStatus_InvalidStatus(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	err := roleService.UpdateStatus(ctx, 1, 2)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "角色状态值无效")
}

func TestRoleGetOptions_Success(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	expectedOptions := []vo.Option{
		{Label: "管理员", Value: "1"},
		{Label: "普通用户", Value: "2"},
	}

	mockRoleRepo.FindOptionsFunc = func(ctx context.Context) ([]vo.Option, error) {
		return expectedOptions, nil
	}

	options, err := roleService.GetOptions(ctx)

	assert.NoError(t, err)
	assert.Len(t, options, 2)
	assert.Equal(t, "管理员", options[0].Label)
}

func TestRoleGetFormData_Success(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	expectedForm := &bo.RoleFormBO{
		ID:        func(i int64) *int64 { return &i }(1),
		Name:      "管理员",
		Code:      "ADMIN",
		Sort:      1,
		Status:    1,
		DataScope: 0,
	}

	mockRoleRepo.GetFormDataFunc = func(ctx context.Context, roleID int64) (*bo.RoleFormBO, error) {
		assert.Equal(t, int64(1), roleID)
		return expectedForm, nil
	}

	formData, err := roleService.GetFormData(ctx, 1)

	assert.NoError(t, err)
	assert.NotNil(t, formData)
	assert.Equal(t, "管理员", formData.Name)
	assert.Equal(t, "ADMIN", formData.Code)
}

func TestRoleAssignMenus_Success(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	mockRoleRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysRole, error) {
		return &model.SysRole{
			BaseModel: model.BaseModel{ID: id},
			Code:      "TEST_ROLE",
			Name:      "测试角色",
		}, nil
	}

	mockRoleRepo.AssignMenusFunc = func(ctx context.Context, roleID int64, menuIDs []int64) error {
		assert.Equal(t, int64(1), roleID)
		assert.Equal(t, []int64{1, 2, 3}, menuIDs)
		return nil
	}

	err := roleService.AssignMenus(ctx, 1, []int64{1, 2, 3})

	assert.NoError(t, err)
}

func TestRoleAssignMenus_RoleNotFound(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	mockRoleRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysRole, error) {
		return nil, nil
	}

	err := roleService.AssignMenus(ctx, 999, []int64{1, 2, 3})

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "角色不存在")
}

func TestRoleGetMenuIDs_Success(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	mockRoleRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysRole, error) {
		return &model.SysRole{
			BaseModel: model.BaseModel{ID: id},
			Code:      "TEST_ROLE",
			Name:      "测试角色",
		}, nil
	}

	mockRoleRepo.GetMenuIDsFunc = func(ctx context.Context, roleID int64) ([]int64, error) {
		assert.Equal(t, int64(1), roleID)
		return []int64{1, 2, 3}, nil
	}

	menuIDs, err := roleService.GetMenuIDs(ctx, 1)

	assert.NoError(t, err)
	assert.Equal(t, []int64{1, 2, 3}, menuIDs)
}

func TestRoleGetMenuIDs_RoleNotFound(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	mockRoleRepo.FindByIDFunc = func(ctx context.Context, id int64) (*model.SysRole, error) {
		return nil, nil
	}

	_, err := roleService.GetMenuIDs(ctx, 999)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "角色不存在")
}

func TestRoleRepositoryError_Propagation(t *testing.T) {
	ctx := context.Background()
	mockRoleRepo := new(mock_repository.MockRoleRepository)
	mockMenuRepo := new(mock_repository.MockMenuRepository)
	roleService := NewRoleService(mockRoleRepo, mockMenuRepo)

	dbError := errors.New("database connection failed")

	mockRoleRepo.ExistsByCodeFunc = func(ctx context.Context, code string, excludeID ...int64) (bool, error) {
		return false, dbError
	}

	err := roleService.Create(ctx, &bo.RoleFormBO{Name: "测试", Code: "TEST"})

	assert.Error(t, err)
	assert.Equal(t, dbError, err)
}

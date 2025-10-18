package test

import (
	"fmt"
	"testing"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/stretchr/testify/assert"
)

func TestRoleService_GetRolePage(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	roleService := &service.RoleService{}

	// 测试用例1: 正常分页查询
	t.Run("NormalPagination", func(t *testing.T) {
		// 准备测试数据
		testRole := model.SysRole{
			Name:      "test_role_page",
			Code:      "TEST_ROLE_PAGE",
			Sort:      1,
			Status:    1,
			DataScope: 1,
			Deleted:   0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})

		// 插入测试角色
		result := global.DB.Create(&testRole)
		assert.NoError(t, result.Error)

		// 执行查询
		queryParams := query.RolePageQuery{
			PageNum:  1,
			PageSize: 10,
		}
		pageResult, err := roleService.GetRolePage(queryParams)

		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, pageResult)
		assert.GreaterOrEqual(t, pageResult.Total, int64(1))
		assert.NotEmpty(t, pageResult.List)

		// 清理测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})
	})

	// 测试用例2: 带关键字查询
	t.Run("KeywordSearch", func(t *testing.T) {
		// 准备测试数据
		testRole := model.SysRole{
			Name:      "test_keyword_search_role",
			Code:      "TEST_KEYWORD_SEARCH_ROLE",
			Sort:      1,
			Status:    1,
			DataScope: 1,
			Deleted:   0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})

		// 插入测试角色
		result := global.DB.Create(&testRole)
		assert.NoError(t, result.Error)

		// 执行查询
		queryParams := query.RolePageQuery{
			Keywords: "test_keyword_search_role",
			PageNum:  1,
			PageSize: 10,
		}
		pageResult, err := roleService.GetRolePage(queryParams)

		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, pageResult)
		assert.GreaterOrEqual(t, pageResult.Total, int64(1))
		assert.NotEmpty(t, pageResult.List)

		// 清理测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})
	})
}

func TestRoleService_ListRoleOptions(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	roleService := &service.RoleService{}

	// 测试用例1: 正常获取下拉列表
	t.Run("NormalListOptions", func(t *testing.T) {
		// 准备测试数据
		testRole := model.SysRole{
			Name:      "test_role_options",
			Code:      "TEST_ROLE_OPTIONS",
			Sort:      1,
			Status:    1,
			DataScope: 1,
			Deleted:   0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})

		// 插入测试角色
		result := global.DB.Create(&testRole)
		assert.NoError(t, result.Error)

		// 获取下拉列表
		options, err := roleService.ListRoleOptions()

		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, options)
		assert.NotEmpty(t, options)

		// 清理测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})
	})
}

func TestRoleService_SaveRole(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	roleService := &service.RoleService{}

	// 测试用例1: 正常新增角色
	t.Run("NormalAddRole", func(t *testing.T) {
		// 准备角色表单数据
		roleFormBO := bo.RoleFormBO{
			Name:      "test_add_role",
			Code:      "TEST_ADD_ROLE",
			Sort:      1,
			Status:    1,
			DataScope: 1,
		}

		// 清理可能存在的测试数据
		global.DB.Where("code = ?", roleFormBO.Code).Delete(&model.SysRole{})

		// 保存角色
		err := roleService.SaveRole(roleFormBO)

		// 验证结果
		assert.NoError(t, err)

		// 验证角色是否真的插入数据库
		var savedRole model.SysRole
		result := global.DB.Where("code = ?", roleFormBO.Code).First(&savedRole)
		assert.NoError(t, result.Error)
		assert.Equal(t, roleFormBO.Name, savedRole.Name)
		assert.Equal(t, roleFormBO.Code, savedRole.Code)
		assert.Equal(t, roleFormBO.Sort, savedRole.Sort)

		// 清理测试数据
		global.DB.Where("code = ?", roleFormBO.Code).Delete(&model.SysRole{})
	})

	// 测试用例2: 正常更新角色
	t.Run("NormalUpdateRole", func(t *testing.T) {
		// 创建测试角色
		testRole := model.SysRole{
			Name:      "test_update_role",
			Code:      "TEST_UPDATE_ROLE",
			Sort:      1,
			Status:    1,
			DataScope: 1,
			Deleted:   0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})

		// 插入测试角色
		result := global.DB.Create(&testRole)
		assert.NoError(t, result.Error)

		// 准备更新数据
		id := testRole.ID
		roleFormBO := bo.RoleFormBO{
			ID:        &id,
			Name:      "test_updated_role",
			Code:      "TEST_UPDATED_ROLE",
			Sort:      2,
			Status:    0,
			DataScope: 2,
		}

		// 更新角色
		err := roleService.SaveRole(roleFormBO)

		// 验证结果
		assert.NoError(t, err)

		// 验证角色是否真的更新
		var updatedRole model.SysRole
		result = global.DB.Where("id = ?", testRole.ID).First(&updatedRole)
		assert.NoError(t, result.Error)
		assert.Equal(t, roleFormBO.Name, updatedRole.Name)
		assert.Equal(t, roleFormBO.Code, updatedRole.Code)
		assert.Equal(t, roleFormBO.Sort, updatedRole.Sort)
		assert.Equal(t, roleFormBO.Status, updatedRole.Status)

		// 清理测试数据
		global.DB.Where("id = ?", testRole.ID).Delete(&model.SysRole{})
	})

	// 测试用例3: 角色名称或编码已存在
	t.Run("DuplicateRole", func(t *testing.T) {
		// 创建测试角色
		testRole := model.SysRole{
			Name:      "test_duplicate_role",
			Code:      "TEST_DUPLICATE_ROLE",
			Sort:      1,
			Status:    1,
			DataScope: 1,
			Deleted:   0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})

		// 插入测试角色
		result := global.DB.Create(&testRole)
		assert.NoError(t, result.Error)

		// 尝试保存相同编码的角色
		roleFormBO := bo.RoleFormBO{
			Name:      "another_test_role",
			Code:      "TEST_DUPLICATE_ROLE",
			Sort:      2,
			Status:    1,
			DataScope: 1,
		}

		err := roleService.SaveRole(roleFormBO)

		// 验证结果
		assert.Error(t, err)
		assert.Equal(t, "角色名称或角色编码已存在，请修改后重试！", err.Error())

		// 清理测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})
	})
}

func TestRoleService_GetRoleForm(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	roleService := &service.RoleService{}

	// 测试用例1: 角色不存在
	t.Run("RoleNotFound", func(t *testing.T) {
		roleFormBO, err := roleService.GetRoleForm(999999)
		assert.Error(t, err)
		assert.Equal(t, "角色不存在", err.Error())
		assert.Equal(t, bo.RoleFormBO{}, roleFormBO)
	})

	// 测试用例2: 角色存在
	t.Run("RoleExists", func(t *testing.T) {
		// 创建测试角色
		testRole := model.SysRole{
			Name:      "test_get_form_role",
			Code:      "TEST_GET_FORM_ROLE",
			Sort:      1,
			Status:    1,
			DataScope: 1,
			Deleted:   0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})

		// 插入测试角色
		result := global.DB.Create(&testRole)
		assert.NoError(t, result.Error)

		// 获取表单数据
		roleFormBO, err := roleService.GetRoleForm(testRole.ID)

		// 验证结果
		assert.NoError(t, err)
		assert.Equal(t, testRole.ID, *roleFormBO.ID)
		assert.Equal(t, testRole.Name, roleFormBO.Name)
		assert.Equal(t, testRole.Code, roleFormBO.Code)
		assert.Equal(t, testRole.Sort, roleFormBO.Sort)

		// 清理测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})
	})
}

func TestRoleService_UpdateRoleStatus(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	roleService := &service.RoleService{}

	// 测试用例1: 正常更新角色状态
	t.Run("NormalUpdateStatus", func(t *testing.T) {
		// 创建测试角色
		testRole := model.SysRole{
			Name:      "test_update_status_role",
			Code:      "TEST_UPDATE_STATUS_ROLE",
			Sort:      1,
			Status:    1,
			DataScope: 1,
			Deleted:   0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})

		// 插入测试角色
		result := global.DB.Create(&testRole)
		assert.NoError(t, result.Error)

		// 更新角色状态
		err := roleService.UpdateRoleStatus(testRole.ID, 0)

		// 验证结果
		assert.NoError(t, err)

		// 验证角色状态是否真的更新
		var updatedRole model.SysRole
		result = global.DB.Where("id = ?", testRole.ID).First(&updatedRole)
		assert.NoError(t, result.Error)
		assert.Equal(t, int8(0), updatedRole.Status)

		// 清理测试数据
		global.DB.Where("id = ?", testRole.ID).Delete(&model.SysRole{})
	})

	// 测试用例2: 角色不存在
	t.Run("RoleNotFound", func(t *testing.T) {
		err := roleService.UpdateRoleStatus(999999, 0)
		assert.Error(t, err)
		assert.Equal(t, "角色不存在", err.Error())
	})
}

func TestRoleService_DeleteRoles(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	roleService := &service.RoleService{}

	// 测试用例1: 正常删除角色
	t.Run("NormalDeleteRoles", func(t *testing.T) {
		// 创建测试角色
		testRole := model.SysRole{
			Name:      "test_delete_role",
			Code:      "TEST_DELETE_ROLE",
			Sort:      1,
			Status:    1,
			DataScope: 1,
			Deleted:   0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})

		// 插入测试角色
		result := global.DB.Create(&testRole)
		assert.NoError(t, result.Error)

		// 删除角色
		err := roleService.DeleteRoles(fmt.Sprintf("%d", testRole.ID))

		// 验证结果
		assert.NoError(t, err)

		// 验证角色是否真的被逻辑删除
		var deletedRole model.SysRole
		result = global.DB.Unscoped().Where("id = ?", testRole.ID).First(&deletedRole)
		assert.NoError(t, result.Error)
		assert.Equal(t, int8(1), deletedRole.Deleted)

		// 清理测试数据
		global.DB.Where("id = ?", testRole.ID).Delete(&model.SysRole{})
	})

	// 测试用例2: 删除多个角色
	t.Run("DeleteMultipleRoles", func(t *testing.T) {
		// 创建测试角色1
		testRole1 := model.SysRole{
			Name:      "test_delete_role_1",
			Code:      "TEST_DELETE_ROLE_1",
			Sort:      1,
			Status:    1,
			DataScope: 1,
			Deleted:   0,
		}

		// 创建测试角色2
		testRole2 := model.SysRole{
			Name:      "test_delete_role_2",
			Code:      "TEST_DELETE_ROLE_2",
			Sort:      1,
			Status:    1,
			DataScope: 1,
			Deleted:   0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("code IN ?", []string{testRole1.Code, testRole2.Code}).Delete(&model.SysRole{})

		// 插入测试角色
		result1 := global.DB.Create(&testRole1)
		assert.NoError(t, result1.Error)
		result2 := global.DB.Create(&testRole2)
		assert.NoError(t, result2.Error)

		// 删除角色
		ids := fmt.Sprintf("%d,%d", testRole1.ID, testRole2.ID)
		err := roleService.DeleteRoles(ids)

		// 验证结果
		assert.NoError(t, err)

		// 验证角色是否真的被逻辑删除
		var deletedRole1 model.SysRole
		result := global.DB.Unscoped().Where("id = ?", testRole1.ID).First(&deletedRole1)
		assert.NoError(t, result.Error)
		assert.Equal(t, int8(1), deletedRole1.Deleted)

		var deletedRole2 model.SysRole
		result = global.DB.Unscoped().Where("id = ?", testRole2.ID).First(&deletedRole2)
		assert.NoError(t, result.Error)
		assert.Equal(t, int8(1), deletedRole2.Deleted)

		// 清理测试数据
		global.DB.Where("id IN ?", []int64{testRole1.ID, testRole2.ID}).Delete(&model.SysRole{})
	})
}

func TestRoleService_GetRoleMenuIds(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	roleService := &service.RoleService{}

	// 测试用例1: 角色不存在
	t.Run("RoleNotFound", func(t *testing.T) {
		menuIds, err := roleService.GetRoleMenuIds(999999)
		assert.Error(t, err)
		assert.Equal(t, "角色不存在", err.Error())
		assert.Empty(t, menuIds)
	})

	// 测试用例2: 角色存在但无菜单
	t.Run("RoleExistsNoMenus", func(t *testing.T) {
		// 创建测试角色
		testRole := model.SysRole{
			Name:      "test_get_menu_ids_role",
			Code:      "TEST_GET_MENU_IDS_ROLE",
			Sort:      1,
			Status:    1,
			DataScope: 1,
			Deleted:   0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})

		// 插入测试角色
		result := global.DB.Create(&testRole)
		assert.NoError(t, result.Error)

		// 获取角色菜单ID集合
		menuIds, err := roleService.GetRoleMenuIds(testRole.ID)

		// 验证结果
		assert.NoError(t, err)
		assert.Empty(t, menuIds)

		// 清理测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})
	})
}

func TestRoleService_AssignMenusToRole(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	roleService := &service.RoleService{}

	// 测试用例1: 角色不存在
	t.Run("RoleNotFound", func(t *testing.T) {
		err := roleService.AssignMenusToRole(999999, []int64{1, 2, 3})
		assert.Error(t, err)
		assert.Equal(t, "角色不存在", err.Error())
	})

	// 测试用例2: 正常分配菜单给角色
	t.Run("NormalAssignMenus", func(t *testing.T) {
		// 创建测试角色
		testRole := model.SysRole{
			Name:      "test_assign_menus_role",
			Code:      "TEST_ASSIGN_MENUS_ROLE",
			Sort:      1,
			Status:    1,
			DataScope: 1,
			Deleted:   0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})

		// 插入测试角色
		result := global.DB.Create(&testRole)
		assert.NoError(t, result.Error)

		// 分配菜单给角色
		menuIds := []int64{1, 2, 3}
		err := roleService.AssignMenusToRole(testRole.ID, menuIds)

		// 验证结果
		assert.NoError(t, err)

		// 验证菜单是否真的分配给了角色
		var roleMenus []model.SysRoleMenu
		result = global.DB.Where("role_id = ?", testRole.ID).Find(&roleMenus)
		assert.NoError(t, result.Error)
		assert.Equal(t, len(menuIds), len(roleMenus))

		// 清理测试数据
		global.DB.Where("role_id = ?", testRole.ID).Delete(&model.SysRoleMenu{})
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})
	})
}

func TestRoleService_GetMaximumDataScope(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	roleService := &service.RoleService{}

	// 测试用例1: 空角色列表
	t.Run("EmptyRoles", func(t *testing.T) {
		dataScope, err := roleService.GetMaximumDataScope([]string{})
		assert.NoError(t, err)
		assert.Nil(t, dataScope)
	})

	// 测试用例2: 正常获取最大数据权限范围
	t.Run("NormalGetMaxDataScope", func(t *testing.T) {
		// 创建测试角色
		testRole1 := model.SysRole{
			Name:      "test_max_scope_role_1",
			Code:      "TEST_MAX_SCOPE_ROLE_1",
			Sort:      1,
			Status:    1,
			DataScope: 1, // 部门及子部门数据
			Deleted:   0,
		}

		testRole2 := model.SysRole{
			Name:      "test_max_scope_role_2",
			Code:      "TEST_MAX_SCOPE_ROLE_2",
			Sort:      1,
			Status:    1,
			DataScope: 2, // 本部门数据
			Deleted:   0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("code IN ?", []string{testRole1.Code, testRole2.Code}).Delete(&model.SysRole{})

		// 插入测试角色
		result1 := global.DB.Create(&testRole1)
		assert.NoError(t, result1.Error)
		result2 := global.DB.Create(&testRole2)
		assert.NoError(t, result2.Error)

		// 获取最大数据权限范围
		roles := []string{testRole1.Code, testRole2.Code}
		dataScope, err := roleService.GetMaximumDataScope(roles)

		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, dataScope)
		assert.Equal(t, int8(1), *dataScope) // 最小值应该是1

		// 清理测试数据
		global.DB.Where("code IN ?", []string{testRole1.Code, testRole2.Code}).Delete(&model.SysRole{})
	})
}

// ============ 补充测试用例：输入参数验证 ============

func TestRoleService_SaveRole_InputValidation(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	roleService := &service.RoleService{}

	// 测试用例1: 角色名称为空
	t.Run("EmptyName", func(t *testing.T) {
		roleFormBO := bo.RoleFormBO{
			Name:      "",
			Code:      "TEST_EMPTY_NAME",
			Sort:      1,
			Status:    1,
			DataScope: 1,
		}

		err := roleService.SaveRole(roleFormBO)
		assert.Error(t, err)
		assert.Equal(t, "角色名称不能为空", err.Error())
	})

	// 测试用例2: 角色名称为纯空格
	t.Run("WhitespaceName", func(t *testing.T) {
		roleFormBO := bo.RoleFormBO{
			Name:      "   ",
			Code:      "TEST_WHITESPACE_NAME",
			Sort:      1,
			Status:    1,
			DataScope: 1,
		}

		err := roleService.SaveRole(roleFormBO)
		assert.Error(t, err)
		assert.Equal(t, "角色名称不能为空", err.Error())
	})

	// 测试用例3: 角色名称超长（>50字符）
	t.Run("TooLongName", func(t *testing.T) {
		roleFormBO := bo.RoleFormBO{
			Name:      "这是一个非常非常非常非常非常非常非常非常非常非常非常非常非常非常长的角色名称超过五十个字符",
			Code:      "TEST_LONG_NAME",
			Sort:      1,
			Status:    1,
			DataScope: 1,
		}

		err := roleService.SaveRole(roleFormBO)
		assert.Error(t, err)
		assert.Equal(t, "角色名称长度不能超过50个字符", err.Error())
	})

	// 测试用例4: 角色编码为空
	t.Run("EmptyCode", func(t *testing.T) {
		roleFormBO := bo.RoleFormBO{
			Name:      "测试角色",
			Code:      "",
			Sort:      1,
			Status:    1,
			DataScope: 1,
		}

		err := roleService.SaveRole(roleFormBO)
		assert.Error(t, err)
		assert.Equal(t, "角色编码不能为空", err.Error())
	})

	// 测试用例5: 角色编码为纯空格
	t.Run("WhitespaceCode", func(t *testing.T) {
		roleFormBO := bo.RoleFormBO{
			Name:      "测试角色",
			Code:      "   ",
			Sort:      1,
			Status:    1,
			DataScope: 1,
		}

		err := roleService.SaveRole(roleFormBO)
		assert.Error(t, err)
		assert.Equal(t, "角色编码不能为空", err.Error())
	})

	// 测试用例6: 角色编码超长（>50字符）
	t.Run("TooLongCode", func(t *testing.T) {
		roleFormBO := bo.RoleFormBO{
			Name:      "测试角色",
			Code:      "TEST_VERY_VERY_VERY_VERY_VERY_VERY_VERY_LONG_CODE_EXCEEDS_FIFTY_CHARACTERS",
			Sort:      1,
			Status:    1,
			DataScope: 1,
		}

		err := roleService.SaveRole(roleFormBO)
		assert.Error(t, err)
		assert.Equal(t, "角色编码长度不能超过50个字符", err.Error())
	})

	// 测试用例7: 角色状态值非法（非0非1）
	t.Run("InvalidStatus", func(t *testing.T) {
		roleFormBO := bo.RoleFormBO{
			Name:      "测试角色",
			Code:      "TEST_INVALID_STATUS",
			Sort:      1,
			Status:    2, // 非法状态值
			DataScope: 1,
		}

		err := roleService.SaveRole(roleFormBO)
		assert.Error(t, err)
		assert.Equal(t, "角色状态值无效，必须为0或1", err.Error())
	})
}

func TestRoleService_UpdateRoleStatus_InputValidation(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	roleService := &service.RoleService{}

	// 测试用例1: 状态值非法（非0非1）
	t.Run("InvalidStatus", func(t *testing.T) {
		err := roleService.UpdateRoleStatus(1, 2)
		assert.Error(t, err)
		assert.Equal(t, "角色状态值无效，必须为0或1", err.Error())
	})

	// 测试用例2: 状态值为负数
	t.Run("NegativeStatus", func(t *testing.T) {
		err := roleService.UpdateRoleStatus(1, -1)
		assert.Error(t, err)
		assert.Equal(t, "角色状态值无效，必须为0或1", err.Error())
	})
}

// ============ 补充测试用例：边界条件 ============

func TestRoleService_GetRolePage_BoundaryConditions(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	roleService := &service.RoleService{}

	// 测试用例1: 分页参数为0
	t.Run("ZeroPageParams", func(t *testing.T) {
		queryParams := query.RolePageQuery{
			PageNum:  0,
			PageSize: 0,
		}
		pageResult, err := roleService.GetRolePage(queryParams)

		// 验证结果 - 应该使用默认值
		assert.NoError(t, err)
		assert.Equal(t, int64(1), int64(pageResult.PageNum))
		assert.Equal(t, int64(10), int64(pageResult.PageSize))
	})

	// 测试用例2: 分页参数为负数
	t.Run("NegativePageParams", func(t *testing.T) {
		queryParams := query.RolePageQuery{
			PageNum:  -1,
			PageSize: -1,
		}
		pageResult, err := roleService.GetRolePage(queryParams)

		// 验证结果 - 应该使用默认值
		assert.NoError(t, err)
		assert.Equal(t, int64(1), int64(pageResult.PageNum))
		assert.Equal(t, int64(10), int64(pageResult.PageSize))
	})

	// 测试用例3: 关键字为空字符串
	t.Run("EmptyKeyword", func(t *testing.T) {
		queryParams := query.RolePageQuery{
			Keywords: "",
			PageNum:  1,
			PageSize: 10,
		}
		pageResult, err := roleService.GetRolePage(queryParams)

		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, pageResult)
	})

	// 测试用例4: 关键字包含特殊字符
	t.Run("SpecialCharKeyword", func(t *testing.T) {
		queryParams := query.RolePageQuery{
			Keywords: "%_\\",
			PageNum:  1,
			PageSize: 10,
		}
		pageResult, err := roleService.GetRolePage(queryParams)

		// 验证结果 - 不应该报错
		assert.NoError(t, err)
		assert.NotNil(t, pageResult)
	})
}

func TestRoleService_DeleteRoles_InputValidation(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	roleService := &service.RoleService{}

	// 测试用例1: ID为空字符串
	t.Run("EmptyIds", func(t *testing.T) {
		err := roleService.DeleteRoles("")
		assert.Error(t, err)
		assert.Equal(t, "删除的角色ID不能为空", err.Error())
	})

	// 测试用例2: ID格式不正确
	t.Run("InvalidIdFormat", func(t *testing.T) {
		err := roleService.DeleteRoles("abc,def")
		assert.Error(t, err)
		assert.Equal(t, "角色ID格式不正确", err.Error())
	})

	// 测试用例3: 部分角色不存在
	t.Run("PartialRoleNotFound", func(t *testing.T) {
		err := roleService.DeleteRoles("999998,999999")
		assert.Error(t, err)
		assert.Equal(t, "部分角色不存在", err.Error())
	})
}

// ============ 补充测试用例：角色关联用户时删除失败 ============

func TestRoleService_DeleteRoles_WithUserAssigned(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	roleService := &service.RoleService{}

	// 测试用例: 角色已分配用户时无法删除
	t.Run("RoleWithUserAssigned", func(t *testing.T) {
		// 创建测试角色
		testRole := model.SysRole{
			Name:      "test_role_with_user",
			Code:      "TEST_ROLE_WITH_USER",
			Sort:      1,
			Status:    1,
			DataScope: 1,
			Deleted:   0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})
		global.DB.Where("role_id IN (SELECT id FROM sys_role WHERE code = ?)", testRole.Code).Delete(&model.SysUserRole{})

		// 插入测试角色
		result := global.DB.Create(&testRole)
		assert.NoError(t, result.Error)

		// 创建用户角色关联（模拟角色已分配给用户）
		userRole := model.SysUserRole{
			UserID: 1,
			RoleID: testRole.ID,
		}
		result = global.DB.Create(&userRole)
		assert.NoError(t, result.Error)

		// 尝试删除角色
		err := roleService.DeleteRoles(fmt.Sprintf("%d", testRole.ID))

		// 验证结果 - 应该删除失败
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "已分配用户")

		// 清理测试数据
		global.DB.Where("role_id = ?", testRole.ID).Delete(&model.SysUserRole{})
		global.DB.Where("code = ?", testRole.Code).Delete(&model.SysRole{})
	})
}

// ============ 补充测试用例：批量删除性能优化验证 ============

func TestRoleService_DeleteRoles_BatchOptimization(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	roleService := &service.RoleService{}

	// 测试用例: 批量删除多个角色（验证N+1查询优化）
	t.Run("BatchDeleteMultipleRoles", func(t *testing.T) {
		// 创建5个测试角色
		testRoles := make([]model.SysRole, 5)
		roleIds := make([]int64, 5)
		roleCodes := make([]string, 5)

		for i := 0; i < 5; i++ {
			testRoles[i] = model.SysRole{
				Name:      fmt.Sprintf("test_batch_delete_role_%d", i),
				Code:      fmt.Sprintf("TEST_BATCH_DELETE_ROLE_%d", i),
				Sort:      1,
				Status:    1,
				DataScope: 1,
				Deleted:   0,
			}
			roleCodes[i] = testRoles[i].Code
		}

		// 清理可能存在的测试数据
		global.DB.Where("code IN ?", roleCodes).Delete(&model.SysRole{})

		// 批量插入测试角色
		for i := 0; i < 5; i++ {
			result := global.DB.Create(&testRoles[i])
			assert.NoError(t, result.Error)
			roleIds[i] = testRoles[i].ID
		}

		// 批量删除角色
		ids := fmt.Sprintf("%d,%d,%d,%d,%d", roleIds[0], roleIds[1], roleIds[2], roleIds[3], roleIds[4])
		err := roleService.DeleteRoles(ids)

		// 验证结果
		assert.NoError(t, err)

		// 验证所有角色都被逻辑删除
		for _, roleId := range roleIds {
			var deletedRole model.SysRole
			result := global.DB.Unscoped().Where("id = ?", roleId).First(&deletedRole)
			assert.NoError(t, result.Error)
			assert.Equal(t, int8(1), deletedRole.Deleted)
		}

		// 清理测试数据
		global.DB.Where("id IN ?", roleIds).Delete(&model.SysRole{})
	})
}

// ============ 补充测试用例：特殊字符处理 ============

func TestRoleService_SaveRole_SpecialCharacters(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	roleService := &service.RoleService{}

	// 测试用例: 角色名称包含特殊字符
	t.Run("SpecialCharactersInName", func(t *testing.T) {
		roleFormBO := bo.RoleFormBO{
			Name:      "测试角色<>\"'&",
			Code:      "TEST_SPECIAL_CHAR_ROLE",
			Sort:      1,
			Status:    1,
			DataScope: 1,
		}

		// 清理可能存在的测试数据
		global.DB.Where("code = ?", roleFormBO.Code).Delete(&model.SysRole{})

		// 保存角色
		err := roleService.SaveRole(roleFormBO)

		// 验证结果 - 应该能正常保存
		assert.NoError(t, err)

		// 验证数据
		var savedRole model.SysRole
		result := global.DB.Where("code = ?", roleFormBO.Code).First(&savedRole)
		assert.NoError(t, result.Error)
		assert.Equal(t, roleFormBO.Name, savedRole.Name)

		// 清理测试数据
		global.DB.Where("code = ?", roleFormBO.Code).Delete(&model.SysRole{})
	})
}

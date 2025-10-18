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

func TestDeptService_ListDepartments(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	deptService := &service.DeptService{}

	// 测试用例1: 正常获取部门列表
	t.Run("NormalListDepartments", func(t *testing.T) {
		// 准备测试数据
		testDept := model.SysDept{
			Name:     "test_dept_list",
			ParentID: 0,
			TreePath: "0",
			Sort:     1,
			Status:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", testDept.Name).Delete(&model.SysDept{})

		// 插入测试部门
		result := global.DB.Create(&testDept)
		assert.NoError(t, result.Error)

		// 执行查询
		queryParams := query.DeptQuery{}
		deptVOs, err := deptService.ListDepartments(queryParams)
		
		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, deptVOs)
		assert.NotEmpty(t, deptVOs)

		// 清理测试数据
		global.DB.Where("name = ?", testDept.Name).Delete(&model.SysDept{})
	})

	// 测试用例2: 带关键字查询
	t.Run("KeywordSearch", func(t *testing.T) {
		// 准备测试数据
		testDept := model.SysDept{
			Name:     "test_keyword_search_dept",
			ParentID: 0,
			TreePath: "0",
			Sort:     1,
			Status:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", testDept.Name).Delete(&model.SysDept{})

		// 插入测试部门
		result := global.DB.Create(&testDept)
		assert.NoError(t, result.Error)

		// 执行查询
		queryParams := query.DeptQuery{
			Keywords: "test_keyword_search_dept",
		}
		deptVOs, err := deptService.ListDepartments(queryParams)
		
		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, deptVOs)
		assert.NotEmpty(t, deptVOs)

		// 清理测试数据
		global.DB.Where("name = ?", testDept.Name).Delete(&model.SysDept{})
	})

	// 测试用例3: 带状态查询
	t.Run("StatusSearch", func(t *testing.T) {
		// 准备测试数据
		testDept := model.SysDept{
			Name:     "test_status_search_dept",
			ParentID: 0,
			TreePath: "0",
			Sort:     1,
			Status:   0, // 禁用状态
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", testDept.Name).Delete(&model.SysDept{})

		// 插入测试部门
		result := global.DB.Create(&testDept)
		assert.NoError(t, result.Error)

		// 执行查询
		status := 0
		queryParams := query.DeptQuery{
			Status: &status,
		}
		deptVOs, err := deptService.ListDepartments(queryParams)
		
		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, deptVOs)

		// 清理测试数据
		global.DB.Where("name = ?", testDept.Name).Delete(&model.SysDept{})
	})
}

func TestDeptService_ListDeptOptions(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	deptService := &service.DeptService{}

	// 测试用例1: 正常获取部门下拉选项
	t.Run("NormalListOptions", func(t *testing.T) {
		// 准备测试数据
		testDept := model.SysDept{
			Name:     "test_dept_options",
			ParentID: 0,
			TreePath: "0",
			Sort:     1,
			Status:   1, // 启用状态
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", testDept.Name).Delete(&model.SysDept{})

		// 插入测试部门
		result := global.DB.Create(&testDept)
		assert.NoError(t, result.Error)

		// 获取部门下拉选项
		options, err := deptService.ListDeptOptions()
		
		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, options)

		// 清理测试数据
		global.DB.Where("name = ?", testDept.Name).Delete(&model.SysDept{})
	})

	// 测试用例2: 无启用部门
	t.Run("NoEnabledDepts", func(t *testing.T) {
		// 准备测试数据
		testDept := model.SysDept{
			Name:     "test_dept_options_disabled",
			ParentID: 0,
			TreePath: "0",
			Sort:     1,
			Status:   0, // 禁用状态
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", testDept.Name).Delete(&model.SysDept{})

		// 插入测试部门
		result := global.DB.Create(&testDept)
		assert.NoError(t, result.Error)

		// 获取部门下拉选项
		options, err := deptService.ListDeptOptions()
		
		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, options)

		// 清理测试数据
		global.DB.Where("name = ?", testDept.Name).Delete(&model.SysDept{})
	})
}

func TestDeptService_SaveDept(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	deptService := &service.DeptService{}

	// 测试用例1: 正常新增部门
	t.Run("NormalSaveDept", func(t *testing.T) {
		// 准备部门表单数据
		deptFormBO := bo.DeptFormBO{
			Name:     "test_save_dept",
			ParentID: 0,
			Status:   1,
			Sort:     1,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", deptFormBO.Name).Delete(&model.SysDept{})

		// 保存部门
		id, err := deptService.SaveDept(deptFormBO)
		
		// 验证结果
		assert.NoError(t, err)
		assert.Greater(t, id, int64(0))

		// 验证部门是否真的插入数据库
		var savedDept model.SysDept
		result := global.DB.Where("name = ?", deptFormBO.Name).First(&savedDept)
		assert.NoError(t, result.Error)
		assert.Equal(t, deptFormBO.Name, savedDept.Name)
		assert.Equal(t, deptFormBO.ParentID, savedDept.ParentID)
		assert.Equal(t, deptFormBO.Sort, savedDept.Sort)

		// 清理测试数据
		global.DB.Where("name = ?", deptFormBO.Name).Delete(&model.SysDept{})
	})

	// 测试用例2: 部门名称已存在
	t.Run("DuplicateDeptName", func(t *testing.T) {
		// 创建测试部门
		testDept := model.SysDept{
			Name:     "test_duplicate_dept",
			ParentID: 0,
			TreePath: "0",
			Sort:     1,
			Status:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", testDept.Name).Delete(&model.SysDept{})

		// 插入测试部门
		result := global.DB.Create(&testDept)
		assert.NoError(t, result.Error)

		// 尝试保存相同名称的部门
		deptFormBO := bo.DeptFormBO{
			Name:     "test_duplicate_dept",
			ParentID: 0,
			Status:   1,
			Sort:     2,
		}

		id, err := deptService.SaveDept(deptFormBO)
		
		// 验证结果
		assert.Error(t, err)
		assert.Equal(t, "部门名称已存在", err.Error())
		assert.Equal(t, int64(0), id)

		// 清理测试数据
		global.DB.Where("name = ?", testDept.Name).Delete(&model.SysDept{})
	})
}

func TestDeptService_UpdateDept(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	deptService := &service.DeptService{}

	// 测试用例1: 正常更新部门
	t.Run("NormalUpdateDept", func(t *testing.T) {
		// 创建测试部门
		testDept := model.SysDept{
			Name:     "test_update_dept",
			ParentID: 0,
			TreePath: "0",
			Sort:     1,
			Status:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", testDept.Name).Delete(&model.SysDept{})

		// 插入测试部门
		result := global.DB.Create(&testDept)
		assert.NoError(t, result.Error)

		// 准备更新数据
		deptFormBO := bo.DeptFormBO{
			Name:     "test_updated_dept",
			ParentID: 0,
			Status:   0,
			Sort:     2,
		}

		// 更新部门
		id, err := deptService.UpdateDept(testDept.ID, deptFormBO)
		
		// 验证结果
		assert.NoError(t, err)
		assert.Equal(t, testDept.ID, id)

		// 验证部门是否真的更新
		var updatedDept model.SysDept
		result = global.DB.Where("id = ?", testDept.ID).First(&updatedDept)
		assert.NoError(t, result.Error)
		assert.Equal(t, deptFormBO.Name, updatedDept.Name)
		assert.Equal(t, deptFormBO.Status, updatedDept.Status)
		assert.Equal(t, deptFormBO.Sort, updatedDept.Sort)

		// 清理测试数据
		global.DB.Where("id = ?", testDept.ID).Delete(&model.SysDept{})
	})

	// 测试用例2: 部门名称已存在
	t.Run("DuplicateDeptName", func(t *testing.T) {
		// 创建测试部门1
		testDept1 := model.SysDept{
			Name:     "test_duplicate_dept_1",
			ParentID: 0,
			TreePath: "0",
			Sort:     1,
			Status:   1,
			Deleted:  0,
		}

		// 创建测试部门2
		testDept2 := model.SysDept{
			Name:     "test_duplicate_dept_2",
			ParentID: 0,
			TreePath: "0",
			Sort:     1,
			Status:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name IN ?", []string{testDept1.Name, testDept2.Name}).Delete(&model.SysDept{})

		// 插入测试部门
		result1 := global.DB.Create(&testDept1)
		assert.NoError(t, result1.Error)
		result2 := global.DB.Create(&testDept2)
		assert.NoError(t, result2.Error)

		// 尝试将部门2的名称更新为部门1的名称
		deptFormBO := bo.DeptFormBO{
			Name:     "test_duplicate_dept_1",
			ParentID: 0,
			Status:   1,
			Sort:     2,
		}

		id, err := deptService.UpdateDept(testDept2.ID, deptFormBO)
		
		// 验证结果
		assert.Error(t, err)
		assert.Equal(t, "部门名称已存在", err.Error())
		assert.Equal(t, int64(0), id)

		// 清理测试数据
		global.DB.Where("name IN ?", []string{testDept1.Name, testDept2.Name}).Delete(&model.SysDept{})
	})
}

func TestDeptService_DeleteByIds(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	deptService := &service.DeptService{}

	// 测试用例1: 正常删除部门
	t.Run("NormalDeleteDepts", func(t *testing.T) {
		// 创建测试部门
		testDept := model.SysDept{
			Name:     "test_delete_dept",
			ParentID: 0,
			TreePath: "0",
			Sort:     1,
			Status:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", testDept.Name).Delete(&model.SysDept{})

		// 插入测试部门
		result := global.DB.Create(&testDept)
		assert.NoError(t, result.Error)

		// 删除部门
		err := deptService.DeleteByIds(fmt.Sprintf("%d", testDept.ID))
		
		// 验证结果
		assert.NoError(t, err)

		// 验证部门是否真的被删除
		var deletedDept model.SysDept
		result = global.DB.Unscoped().Where("id = ?", testDept.ID).First(&deletedDept)
		assert.NoError(t, result.Error)

		// 清理测试数据
		global.DB.Where("id = ?", testDept.ID).Delete(&model.SysDept{})
	})

	// 测试用例2: 删除多个部门
	t.Run("DeleteMultipleDepts", func(t *testing.T) {
		// 创建测试部门1
		testDept1 := model.SysDept{
			Name:     "test_delete_dept_1",
			ParentID: 0,
			TreePath: "0",
			Sort:     1,
			Status:   1,
			Deleted:  0,
		}

		// 创建测试部门2
		testDept2 := model.SysDept{
			Name:     "test_delete_dept_2",
			ParentID: 0,
			TreePath: "0",
			Sort:     1,
			Status:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name IN ?", []string{testDept1.Name, testDept2.Name}).Delete(&model.SysDept{})

		// 插入测试部门
		result1 := global.DB.Create(&testDept1)
		assert.NoError(t, result1.Error)
		result2 := global.DB.Create(&testDept2)
		assert.NoError(t, result2.Error)

		// 删除部门
		ids := fmt.Sprintf("%d,%d", testDept1.ID, testDept2.ID)
		err := deptService.DeleteByIds(ids)
		
		// 验证结果
		assert.NoError(t, err)

		// 验证部门是否真的被删除
		var deletedDept1 model.SysDept
		result := global.DB.Unscoped().Where("id = ?", testDept1.ID).First(&deletedDept1)
		assert.NoError(t, result.Error)

		var deletedDept2 model.SysDept
		result = global.DB.Unscoped().Where("id = ?", testDept2.ID).First(&deletedDept2)
		assert.NoError(t, result.Error)

		// 清理测试数据
		global.DB.Where("id IN ?", []int64{testDept1.ID, testDept2.ID}).Delete(&model.SysDept{})
	})
}

func TestDeptService_GetDeptForm(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	deptService := &service.DeptService{}

	// 测试用例1: 部门不存在
	t.Run("DeptNotFound", func(t *testing.T) {
		deptFormBO, err := deptService.GetDeptForm(999999)
		assert.Error(t, err)
		assert.Equal(t, "部门不存在", err.Error())
		assert.Equal(t, bo.DeptFormBO{}, deptFormBO)
	})

	// 测试用例2: 部门存在
	t.Run("DeptExists", func(t *testing.T) {
		// 创建测试部门
		testDept := model.SysDept{
			Name:     "test_get_form_dept",
			ParentID: 0,
			TreePath: "0",
			Sort:     1,
			Status:   1,
			Deleted:  0,
		}

		// 清理可能存在的测试数据
		global.DB.Where("name = ?", testDept.Name).Delete(&model.SysDept{})

		// 插入测试部门
		result := global.DB.Create(&testDept)
		assert.NoError(t, result.Error)

		// 获取部门表单数据
		deptFormBO, err := deptService.GetDeptForm(testDept.ID)
		
		// 验证结果
		assert.NoError(t, err)
		assert.NotNil(t, deptFormBO.ID)
		assert.Equal(t, testDept.ID, *deptFormBO.ID)
		assert.Equal(t, testDept.Name, deptFormBO.Name)
		assert.Equal(t, testDept.ParentID, deptFormBO.ParentID)
		assert.Equal(t, testDept.Status, deptFormBO.Status)
		assert.Equal(t, testDept.Sort, deptFormBO.Sort)

		// 清理测试数据
		global.DB.Where("name = ?", testDept.Name).Delete(&model.SysDept{})
	})
}

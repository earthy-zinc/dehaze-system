package test

import (
	"fmt"
	"testing"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/stretchr/testify/suite"
)

// DeptServiceTestSuite 部门服务测试套件
// 使用事务隔离，每个测试方法都在独立事务中运行
type DeptServiceTestSuite struct {
	TransactionTestSuite
	deptService *service.DeptService
}

// SetupSuite 在整个测试套件开始前运行一次
func (s *DeptServiceTestSuite) SetupSuite() {
	// 初始化配置和数据库
	initialize.Viper()
	initialize.Gorm()
	initialize.Redis()

	if global.DB == nil {
		s.T().Fatal("数据库连接失败")
	}

	// 保存原始数据库连接
	s.DB = global.DB

	// 初始化服务
	s.deptService = &service.DeptService{}

	// 确保必要的表已创建
	initialize.Migrate()
}

// TestListDepartments_Normal 测试正常获取部门列表
func (s *DeptServiceTestSuite) TestListDepartments_Normal() {
	// 准备测试数据
	testDept := &model.SysDept{
		Name:     "test_dept_list",
		ParentID: 0,
		TreePath: "0",
		Sort:     1,
		Status:   1,
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testDept))

	// 执行查询
	queryParams := query.DeptQuery{}
	deptVOs, err := s.deptService.ListDepartments(queryParams)

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(deptVOs)
	s.Assert().NotEmpty(deptVOs)

}

// TestListDepartments_KeywordSearch 测试带关键字查询
func (s *DeptServiceTestSuite) TestListDepartments_KeywordSearch() {
	// 准备测试数据
	testDept := &model.SysDept{
		Name:     "test_keyword_search_dept",
		ParentID: 0,
		TreePath: "0",
		Sort:     1,
		Status:   1,
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testDept))

	// 执行查询
	queryParams := query.DeptQuery{
		Keywords: "test_keyword_search_dept",
	}
	deptVOs, err := s.deptService.ListDepartments(queryParams)

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(deptVOs)
	s.Assert().NotEmpty(deptVOs)

}

// TestListDepartments_StatusSearch 测试带状态查询
func (s *DeptServiceTestSuite) TestListDepartments_StatusSearch() {
	// 准备测试数据
	testDept := &model.SysDept{
		Name:     "test_status_search_dept",
		ParentID: 0,
		TreePath: "0",
		Sort:     1,
		Status:   0, // 禁用状态
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testDept))

	// 执行查询
	status := 0
	queryParams := query.DeptQuery{
		Status: &status,
	}
	deptVOs, err := s.deptService.ListDepartments(queryParams)

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(deptVOs)

}

// TestListDeptOptions_Normal 测试正常获取部门下拉选项
func (s *DeptServiceTestSuite) TestListDeptOptions_Normal() {
	// 准备测试数据
	testDept := &model.SysDept{
		Name:     "test_dept_options",
		ParentID: 0,
		TreePath: "0",
		Sort:     1,
		Status:   1, // 启用状态
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testDept))

	// 获取部门下拉选项
	options, err := s.deptService.ListDeptOptions()

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(options)

}

// TestListDeptOptions_NoEnabledDepts 测试无启用部门
func (s *DeptServiceTestSuite) TestListDeptOptions_NoEnabledDepts() {
	// 准备测试数据
	testDept := &model.SysDept{
		Name:     "test_dept_options_disabled",
		ParentID: 0,
		TreePath: "0",
		Sort:     1,
		Status:   0, // 禁用状态
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testDept))

	// 获取部门下拉选项
	options, err := s.deptService.ListDeptOptions()

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(options)

}

// TestSaveDept_Normal 测试正常新增部门
func (s *DeptServiceTestSuite) TestSaveDept_Normal() {
	// 准备部门表单数据
	deptFormBO := bo.DeptFormBO{
		Name:     "test_save_dept",
		ParentID: 0,
		Status:   1,
		Sort:     1,
	}

	// 保存部门
	id, err := s.deptService.SaveDept(deptFormBO)

	// 验证结果
	s.AssertNoError(err)
	s.Assert().Greater(id, int64(0))

	// 验证部门是否真的插入数据库
	var savedDept model.SysDept
	err = s.GetDB().Where("name = ?", deptFormBO.Name).First(&savedDept).Error
	s.AssertNoError(err)
	s.AssertEqual(deptFormBO.Name, savedDept.Name)
	s.AssertEqual(deptFormBO.ParentID, savedDept.ParentID)
	s.AssertEqual(deptFormBO.Sort, savedDept.Sort)

}

// TestSaveDept_DuplicateName 测试部门名称已存在
func (s *DeptServiceTestSuite) TestSaveDept_DuplicateName() {
	// 创建测试部门
	testDept := &model.SysDept{
		Name:     "test_duplicate_dept",
		ParentID: 0,
		TreePath: "0",
		Sort:     1,
		Status:   1,
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testDept))

	// 尝试保存相同名称的部门
	deptFormBO := bo.DeptFormBO{
		Name:     "test_duplicate_dept",
		ParentID: 0,
		Status:   1,
		Sort:     2,
	}

	id, err := s.deptService.SaveDept(deptFormBO)

	// 验证结果
	s.AssertError(err)
	s.AssertEqual("部门名称已存在", err.Error())
	s.AssertEqual(int64(0), id)

}

// TestUpdateDept_Normal 测试正常更新部门
func (s *DeptServiceTestSuite) TestUpdateDept_Normal() {
	// 创建测试部门
	testDept := &model.SysDept{
		Name:     "test_update_dept",
		ParentID: 0,
		TreePath: "0",
		Sort:     1,
		Status:   1,
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testDept))

	// 准备更新数据
	deptFormBO := bo.DeptFormBO{
		Name:     "test_updated_dept",
		ParentID: 0,
		Status:   0,
		Sort:     2,
	}

	// 更新部门
	id, err := s.deptService.UpdateDept(testDept.ID, deptFormBO)

	// 验证结果
	s.AssertNoError(err)
	s.AssertEqual(testDept.ID, id)

	// 验证部门是否真的更新
	var updatedDept model.SysDept
	err = s.GetDB().Where("id = ?", testDept.ID).First(&updatedDept).Error
	s.AssertNoError(err)
	s.AssertEqual(deptFormBO.Name, updatedDept.Name)
	s.AssertEqual(deptFormBO.Status, updatedDept.Status)
	s.AssertEqual(deptFormBO.Sort, updatedDept.Sort)

}

// TestUpdateDept_DuplicateName 测试更新时部门名称已存在
func (s *DeptServiceTestSuite) TestUpdateDept_DuplicateName() {
	// 创建测试部门1
	testDept1 := &model.SysDept{
		Name:     "test_duplicate_dept_1",
		ParentID: 0,
		TreePath: "0",
		Sort:     1,
		Status:   1,
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testDept1))

	// 创建测试部门2
	testDept2 := &model.SysDept{
		Name:     "test_duplicate_dept_2",
		ParentID: 0,
		TreePath: "0",
		Sort:     1,
		Status:   1,
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testDept2))

	// 尝试将部门2的名称更新为部门1的名称
	deptFormBO := bo.DeptFormBO{
		Name:     "test_duplicate_dept_1",
		ParentID: 0,
		Status:   1,
		Sort:     2,
	}

	id, err := s.deptService.UpdateDept(testDept2.ID, deptFormBO)

	// 验证结果
	s.AssertError(err)
	s.AssertEqual("部门名称已存在", err.Error())
	s.AssertEqual(int64(0), id)

}

// TestDeleteByIds_Single 测试正常删除单个部门
func (s *DeptServiceTestSuite) TestDeleteByIds_Single() {
	// 创建测试部门
	testDept := &model.SysDept{
		Name:     "test_delete_dept",
		ParentID: 0,
		TreePath: "0",
		Sort:     1,
		Status:   1,
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testDept))

	// 删除部门
	err := s.deptService.DeleteByIds(fmt.Sprintf("%d", testDept.ID))

	// 验证结果
	s.AssertNoError(err)

	// 验证部门是否真的被删除
	var deletedDept model.SysDept
	err = s.GetDB().Unscoped().Where("id = ?", testDept.ID).First(&deletedDept).Error
	s.AssertNoError(err)

}

// TestDeleteByIds_Multiple 测试删除多个部门
func (s *DeptServiceTestSuite) TestDeleteByIds_Multiple() {
	// 创建测试部门1
	testDept1 := &model.SysDept{
		Name:     "test_delete_dept_1",
		ParentID: 0,
		TreePath: "0",
		Sort:     1,
		Status:   1,
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testDept1))

	// 创建测试部门2
	testDept2 := &model.SysDept{
		Name:     "test_delete_dept_2",
		ParentID: 0,
		TreePath: "0",
		Sort:     1,
		Status:   1,
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testDept2))

	// 删除部门
	ids := fmt.Sprintf("%d,%d", testDept1.ID, testDept2.ID)
	err := s.deptService.DeleteByIds(ids)

	// 验证结果
	s.AssertNoError(err)

	// 验证部门是否真的被删除
	var deletedDept1 model.SysDept
	err = s.GetDB().Unscoped().Where("id = ?", testDept1.ID).First(&deletedDept1).Error
	s.AssertNoError(err)

	var deletedDept2 model.SysDept
	err = s.GetDB().Unscoped().Where("id = ?", testDept2.ID).First(&deletedDept2).Error
	s.AssertNoError(err)

}

// TestGetDeptForm_NotFound 测试部门不存在
func (s *DeptServiceTestSuite) TestGetDeptForm_NotFound() {
	deptFormBO, err := s.deptService.GetDeptForm(999999)
	s.AssertError(err)
	s.AssertEqual("部门不存在", err.Error())
	s.AssertEqual(bo.DeptFormBO{}, deptFormBO)
}

// TestGetDeptForm_Exists 测试部门存在
func (s *DeptServiceTestSuite) TestGetDeptForm_Exists() {
	// 创建测试部门
	testDept := &model.SysDept{
		Name:     "test_get_form_dept",
		ParentID: 0,
		TreePath: "0",
		Sort:     1,
		Status:   1,
		Deleted:  0,
	}
	s.AssertNoError(s.CreateTestData(testDept))

	// 获取部门表单数据
	deptFormBO, err := s.deptService.GetDeptForm(testDept.ID)

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(deptFormBO.ID)
	s.AssertEqual(testDept.ID, *deptFormBO.ID)
	s.AssertEqual(testDept.Name, deptFormBO.Name)
	s.AssertEqual(testDept.ParentID, deptFormBO.ParentID)
	s.AssertEqual(testDept.Status, deptFormBO.Status)
	s.AssertEqual(testDept.Sort, deptFormBO.Sort)

}

// 运行测试套件
func TestDeptService(t *testing.T) {
	suite.Run(t, new(DeptServiceTestSuite))
}

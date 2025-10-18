package test

import (
	"testing"

	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/stretchr/testify/suite"
)

// AlgorithmServiceTestSuite 算法服务测试套件
// 使用事务隔离，每个测试方法都在独立事务中运行
type AlgorithmServiceTestSuite struct {
	TransactionTestSuite
	algorithmService *service.AlgorithmService
}

// SetupSuite 在整个测试套件开始前运行一次
func (s *AlgorithmServiceTestSuite) SetupSuite() {
	// 初始化服务
	s.algorithmService = &service.AlgorithmService{}
}

// TestGetAlgorithmList_Normal 测试正常获取算法列表
func (s *AlgorithmServiceTestSuite) TestGetAlgorithmList_Normal() {
	// 准备测试数据
	testAlgorithm := &model.SysAlgorithm{
		ParentID: 0,
		Type:     "test_type",
		Name:     "test_algorithm_list",
		Path:     "/test/path",
		Status:   1,
		CreateBy: 1,
		UpdateBy: 1,
	}
	s.AssertNoError(s.CreateTestData(testAlgorithm))

	// 执行查询
	queryParams := query.AlgorithmQuery{}
	algorithms, err := s.algorithmService.GetAlgorithmList(queryParams)

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(algorithms)
	s.Assert().NotEmpty(algorithms)

}

// TestGetAlgorithmList_WithKeywords 测试带关键字查询
func (s *AlgorithmServiceTestSuite) TestGetAlgorithmList_WithKeywords() {
	// 准备测试数据
	testAlgorithm1 := &model.SysAlgorithm{
		ParentID: 0,
		Type:     "test_type",
		Name:     "test_algorithm_keyword_1",
		Path:     "/test/path",
		Status:   1,
		CreateBy: 1,
		UpdateBy: 1,
	}
	s.AssertNoError(s.CreateTestData(testAlgorithm1))

	testAlgorithm2 := &model.SysAlgorithm{
		ParentID: 0,
		Type:     "test_type",
		Name:     "another_algorithm",
		Path:     "/test/path",
		Status:   1,
		CreateBy: 1,
		UpdateBy: 1,
	}
	s.AssertNoError(s.CreateTestData(testAlgorithm2))

	// 执行查询
	queryParams := query.AlgorithmQuery{
		Keywords: "test_algorithm_keyword",
	}
	algorithms, err := s.algorithmService.GetAlgorithmList(queryParams)

	// 验证结果
	s.AssertNoError(err)
	s.AssertNotNil(algorithms)
	s.Assert().Equal(1, len(algorithms))
	s.Assert().Equal(testAlgorithm1.Name, algorithms[0].Name)

}

// TestAddAlgorithm_Normal 测试正常添加算法
func (s *AlgorithmServiceTestSuite) TestAddAlgorithm_Normal() {
	algorithmForm := bo.AlgorithmFormBO{
		ParentID: 0,
		Type:     "test_type",
		Name:     "test_add_algorithm",
		Path:     "/test/path",
		Status:   1,
	}

	// 执行添加
	err := s.algorithmService.AddAlgorithm(algorithmForm)

	// 验证结果
	s.AssertNoError(err)

	// 验证数据是否插入
	var algorithm model.SysAlgorithm
	err = s.GetDB().Where("name = ?", algorithmForm.Name).First(&algorithm).Error
	s.AssertNoError(err)
	s.AssertEqual(algorithmForm.Name, algorithm.Name)
	s.AssertEqual(algorithmForm.Type, algorithm.Type)
	s.AssertEqual(algorithmForm.Path, algorithm.Path)

}

// TestAddAlgorithm_WithParent 测试添加带父节点的算法
func (s *AlgorithmServiceTestSuite) TestAddAlgorithm_WithParent() {
	// 先创建父算法
	parentAlgorithm := &model.SysAlgorithm{
		ParentID: 0,
		Type:     "test_type",
		Name:     "test_parent_algorithm",
		Path:     "/test/parent/path",
		Status:   1,
		CreateBy: 1,
		UpdateBy: 1,
	}
	s.AssertNoError(s.CreateTestData(parentAlgorithm))

	// 创建子算法
	algorithmForm := bo.AlgorithmFormBO{
		ParentID: parentAlgorithm.ID,
		Type:     "test_type",
		Name:     "test_child_algorithm",
		Path:     "/test/child/path",
		Status:   1,
	}

	// 执行添加
	err := s.algorithmService.AddAlgorithm(algorithmForm)

	// 验证结果
	s.AssertNoError(err)

	// 验证数据是否插入
	var algorithm model.SysAlgorithm
	err = s.GetDB().Where("name = ?", algorithmForm.Name).First(&algorithm).Error
	s.AssertNoError(err)
	s.AssertEqual(algorithmForm.Name, algorithm.Name)
	s.AssertEqual(algorithmForm.ParentID, algorithm.ParentID)

}

// TestUpdateAlgorithm_Normal 测试正常更新算法
func (s *AlgorithmServiceTestSuite) TestUpdateAlgorithm_Normal() {
	// 准备测试数据
	testAlgorithm := &model.SysAlgorithm{
		ParentID: 0,
		Type:     "test_type",
		Name:     "test_update_algorithm",
		Path:     "/test/path",
		Status:   1,
		CreateBy: 1,
		UpdateBy: 1,
	}
	s.AssertNoError(s.CreateTestData(testAlgorithm))

	// 更新算法
	algorithmForm := bo.AlgorithmFormBO{
		ParentID: 0,
		Type:     "updated_type",
		Name:     "updated_algorithm_name",
		Path:     "/updated/path",
		Status:   1,
	}

	err := s.algorithmService.UpdateAlgorithm(testAlgorithm.ID, algorithmForm)

	// 验证结果
	s.AssertNoError(err)

	// 验证数据是否更新
	var algorithm model.SysAlgorithm
	err = s.GetDB().Where("id = ?", testAlgorithm.ID).First(&algorithm).Error
	s.AssertNoError(err)
	s.AssertEqual(algorithmForm.Name, algorithm.Name)
	s.AssertEqual(algorithmForm.Type, algorithm.Type)
	s.AssertEqual(algorithmForm.Path, algorithm.Path)

}

// TestDeleteAlgorithms_Normal 测试正常删除算法
func (s *AlgorithmServiceTestSuite) TestDeleteAlgorithms_Normal() {
	// 准备测试数据
	testAlgorithm := &model.SysAlgorithm{
		ParentID: 0,
		Type:     "test_type",
		Name:     "test_delete_algorithm",
		Path:     "/test/path",
		Status:   1,
		CreateBy: 1,
		UpdateBy: 1,
	}
	s.AssertNoError(s.CreateTestData(testAlgorithm))

	// 执行删除
	err := s.algorithmService.DeleteAlgorithms([]int64{testAlgorithm.ID})

	// 验证结果
	s.AssertNoError(err)

	// 验证数据是否删除
	var algorithm model.SysAlgorithm
	err = s.GetDB().Where("id = ?", testAlgorithm.ID).First(&algorithm).Error
	s.AssertError(err) // 应该找不到记录

}

// TestDeleteAlgorithms_WithChildren 测试删除有子算法的算法（应该失败）
func (s *AlgorithmServiceTestSuite) TestDeleteAlgorithms_WithChildren() {
	// 准备测试数据
	parentAlgorithm := &model.SysAlgorithm{
		ParentID: 0,
		Type:     "test_type",
		Name:     "test_parent_algorithm_for_delete",
		Path:     "/test/parent/path",
		Status:   1,
		CreateBy: 1,
		UpdateBy: 1,
	}
	s.AssertNoError(s.CreateTestData(parentAlgorithm))

	// 创建子算法
	childAlgorithm := &model.SysAlgorithm{
		ParentID: parentAlgorithm.ID,
		Type:     "test_type",
		Name:     "test_child_algorithm_for_delete",
		Path:     "/test/child/path",
		Status:   1,
		CreateBy: 1,
		UpdateBy: 1,
	}
	s.AssertNoError(s.CreateTestData(childAlgorithm))

	// 执行删除（应该失败）
	err := s.algorithmService.DeleteAlgorithms([]int64{parentAlgorithm.ID})

	// 验证结果
	s.AssertError(err)
	s.AssertEqual("存在子算法，无法删除", err.Error())

}

// 运行测试套件
func TestAlgorithmService(t *testing.T) {
	suite.Run(t, new(AlgorithmServiceTestSuite))
}

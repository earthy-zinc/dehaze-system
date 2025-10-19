package test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/earthyzinc/dehaze-go/model"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/suite"
)

// DatasetItemTestSuite 数据集项测试套件
// 使用事务隔离，每个测试方法都在独立事务中运行
type DatasetItemTestSuite struct {
	BaseTestSuite
}

// TestCreateDatasetItem 测试创建数据集项
func (s *DatasetItemTestSuite) TestCreateDatasetItem() {
	// 自动迁移SysDatasetItem表
	err := s.GetDB().AutoMigrate(&model.SysDatasetItem{})
	s.Require().NoError(err, "自动迁移SysDatasetItem表失败")

	// 创建测试路由
	gin.SetMode(gin.TestMode)
	router := gin.New()

	// 创建测试数据集
	dataset := &model.SysDataset{
		Name:   "测试数据集",
		Status: 1,
	}
	s.Require().NoError(s.CreateTestData(dataset))

	// 创建请求
	req, _ := http.NewRequest("POST", fmt.Sprintf("/api/v1/dataset/item?datasetId=%d", dataset.ID), nil)

	// 创建响应记录器
	resp := httptest.NewRecorder()

	// 执行请求
	router.ServeHTTP(resp, req)

	// 验证响应
	s.Assert().Equal(http.StatusOK, resp.Code)

}

// TestCreateDatasetItemWithName 测试创建带名称的数据集项
func (s *DatasetItemTestSuite) TestCreateDatasetItemWithName() {
	// 自动迁移SysDatasetItem表
	err := s.GetDB().AutoMigrate(&model.SysDatasetItem{})
	s.Require().NoError(err, "自动迁移SysDatasetItem表失败")

	// 创建测试路由
	gin.SetMode(gin.TestMode)
	router := gin.New()

	// 创建测试数据集
	dataset := &model.SysDataset{
		Name:   "测试数据集",
		Status: 1,
	}
	s.Require().NoError(s.CreateTestData(dataset))

	// 创建请求
	req, _ := http.NewRequest("POST", fmt.Sprintf("/api/v1/dataset/item?datasetId=%d&name=测试数据项", dataset.ID), nil)

	// 创建响应记录器
	resp := httptest.NewRecorder()

	// 执行请求
	router.ServeHTTP(resp, req)

	// 验证响应
	s.Assert().Equal(http.StatusOK, resp.Code)

}

// TestUpdateDatasetItem 测试更新数据集项
func (s *DatasetItemTestSuite) TestUpdateDatasetItem() {
	// 自动迁移SysDatasetItem表
	err := s.GetDB().AutoMigrate(&model.SysDatasetItem{})
	s.Require().NoError(err, "自动迁移SysDatasetItem表失败")

	// 创建测试路由
	gin.SetMode(gin.TestMode)
	router := gin.New()

	// 创建测试数据集
	dataset := &model.SysDataset{
		Name:   "测试数据集",
		Status: 1,
	}
	s.Require().NoError(s.CreateTestData(dataset))

	// 先创建一个数据项用于更新
	datasetItem := &model.SysDatasetItem{
		DatasetID: dataset.ID,
		Name:      "原始名称",
	}
	s.Require().NoError(s.CreateTestData(datasetItem))

	// 创建请求
	req, _ := http.NewRequest("PUT", fmt.Sprintf("/api/v1/dataset/item?datasetItemId=%d&name=更新名称", datasetItem.ID), nil)

	// 创建响应记录器
	resp := httptest.NewRecorder()

	// 执行请求
	router.ServeHTTP(resp, req)

	// 验证响应
	s.Assert().Equal(http.StatusOK, resp.Code)

}

// TestDeleteDatasetItem 测试删除数据集项
func (s *DatasetItemTestSuite) TestDeleteDatasetItem() {
	// 自动迁移SysDatasetItem表
	err := s.GetDB().AutoMigrate(&model.SysDatasetItem{})
	s.Require().NoError(err, "自动迁移SysDatasetItem表失败")

	// 创建测试路由
	gin.SetMode(gin.TestMode)
	router := gin.New()

	// 创建测试数据集
	dataset := &model.SysDataset{
		Name:   "测试数据集",
		Status: 1,
	}
	s.Require().NoError(s.CreateTestData(dataset))

	// 先创建一个数据项用于删除
	datasetItem := &model.SysDatasetItem{
		DatasetID: dataset.ID,
		Name:      "待删除项",
	}
	s.Require().NoError(s.CreateTestData(datasetItem))

	// 创建请求
	req, _ := http.NewRequest("DELETE", fmt.Sprintf("/api/v1/dataset/item?datasetItemId=%d", datasetItem.ID), nil)

	// 创建响应记录器
	resp := httptest.NewRecorder()

	// 执行请求
	router.ServeHTTP(resp, req)

	// 验证响应
	s.Assert().Equal(http.StatusOK, resp.Code)

}

// 运行测试套件
func TestDatasetItem(t *testing.T) {
	suite.Run(t, new(DatasetItemTestSuite))
}

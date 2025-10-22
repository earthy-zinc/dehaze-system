package test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/earthyzinc/dehaze-go/api"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/suite"
)

// DatasetServiceTestSuite 数据集服务测试套件
// 使用事务隔离，每个测试方法都在独立事务中运行
type DatasetServiceTestSuite struct {
	BaseTestSuite
	datasetService *service.DatasetService
}

// SetupSuite 在整个测试套件开始前运行一次
func (s *DatasetServiceTestSuite) SetupSuite() {
	// 初始化服务
	s.datasetService = &service.ServiceGroupApp.DatasetService
}

// TestDatasetAPI_CRUD 测试数据集API的增删改查功能
func (s *DatasetServiceTestSuite) TestDatasetAPI_CRUD() {
	// 创建测试路由器
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(gin.Recovery())

	// 注册路由组
	apiGroup := router.Group("/api/v1")
	{
		apiGroup.GET("/dataset", api.ApiGroupApp.SysDatasetApi.GetDatasetList)
		apiGroup.GET("/dataset/options", api.ApiGroupApp.SysDatasetApi.GetDatasetOptions)
		apiGroup.GET("/dataset/:id/form", api.ApiGroupApp.SysDatasetApi.GetDatasetForm)
		apiGroup.POST("/dataset", api.ApiGroupApp.SysDatasetApi.SaveDataset)
		apiGroup.PUT("/dataset/:id", api.ApiGroupApp.SysDatasetApi.UpdateDataset)
		apiGroup.DELETE("/dataset", api.ApiGroupApp.SysDatasetApi.DeleteDatasets)
	}

	// 创建数据集
	datasetForm := bo.DatasetFormBO{
		ParentID:    0,
		Type:        "test_type",
		Name:        "测试数据集",
		Description: "测试用数据集",
		Path:        "/test/path",
		Status:      1,
	}

	jsonValue, _ := json.Marshal(datasetForm)
	req, _ := http.NewRequest("POST", "/api/v1/dataset", bytes.NewBuffer(jsonValue))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	s.Assert().Equal(http.StatusOK, resp.Code)

	// 查询数据集列表
	req, _ = http.NewRequest("GET", "/api/v1/dataset", nil)
	resp = httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	s.Assert().Equal(http.StatusOK, resp.Code)

	// 获取数据集表单数据
	req, _ = http.NewRequest("GET", "/api/v1/dataset/1/form", nil)
	resp = httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	// 更新数据集
	updateDatasetForm := bo.DatasetFormBO{
		ParentID:    0,
		Type:        "test_type_update",
		Name:        "更新后的测试数据集",
		Description: "更新后的测试用数据集",
		Path:        "/test/path/update",
		Status:      1,
	}

	jsonValue, _ = json.Marshal(updateDatasetForm)
	req, _ = http.NewRequest("PUT", "/api/v1/dataset/1", bytes.NewBuffer(jsonValue))
	req.Header.Set("Content-Type", "application/json")
	resp = httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	s.Assert().Equal(http.StatusOK, resp.Code)

}

// TestDatasetAPI_Options 测试数据集下拉选项API
func (s *DatasetServiceTestSuite) TestDatasetAPI_Options() {
	// 创建测试路由器
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(gin.Recovery())

	// 注册路由
	router.GET("/api/v1/dataset/options", api.ApiGroupApp.SysDatasetApi.GetDatasetOptions)

	// 获取数据集下拉选项
	req, _ := http.NewRequest("GET", "/api/v1/dataset/options", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	s.Assert().Equal(http.StatusOK, resp.Code)

}

// TestDatasetAPI_Delete 测试数据集删除API
func (s *DatasetServiceTestSuite) TestDatasetAPI_Delete() {
	// 创建测试路由器
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(gin.Recovery())

	// 注册路由
	router.DELETE("/api/v1/dataset", api.ApiGroupApp.SysDatasetApi.DeleteDatasets)

	// 删除数据集
	req, _ := http.NewRequest("DELETE", "/api/v1/dataset?ids=1", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	s.Assert().Equal(http.StatusOK, resp.Code)

}

// TestDatasetService_CRUD 测试数据集服务的增删改查功能
func (s *DatasetServiceTestSuite) TestDatasetService_CRUD() {
	// 创建数据集
	datasetForm := bo.DatasetFormBO{
		ParentID:    0,
		Type:        "service_test_type",
		Name:        "服务测试数据集",
		Description: "服务测试用数据集",
		Path:        "/service/test/path",
		Status:      1,
	}

	err := s.datasetService.SaveDataset(datasetForm)
	s.AssertNoError(err)

	// 查询数据集列表
	queryParams := struct {
		Keywords string `json:"keywords"`
	}{
		Keywords: "服务测试",
	}

	datasetVOs, err := s.datasetService.GetDatasetList(queryParams)
	s.AssertNoError(err)
	s.Assert().Greater(len(datasetVOs), 0)

	// 获取刚创建的数据集的ID
	var dataset model.SysDataset
	s.GetDB().Where("name = ?", "服务测试数据集").First(&dataset)
	datasetID := dataset.ID

	// 获取数据集表单
	datasetFormBO, err := s.datasetService.GetDatasetForm(datasetID)
	s.AssertNoError(err)
	s.AssertEqual("服务测试数据集", datasetFormBO.Name)

	// 更新数据集
	updateDatasetForm := bo.DatasetFormBO{
		ParentID:    0,
		Type:        "service_test_type_update",
		Name:        "更新后的服务测试数据集",
		Description: "更新后的服务测试用数据集",
		Path:        "/service/test/path/update",
		Status:      1,
	}

	err = s.datasetService.UpdateDataset(datasetID, updateDatasetForm)
	s.AssertNoError(err)

}

// TestDatasetService_Options 测试数据集下拉选项服务
func (s *DatasetServiceTestSuite) TestDatasetService_Options() {
	// 获取数据集下拉选项
	options, err := s.datasetService.GetDatasetOptions()
	s.AssertNoError(err)
	s.Assert().Greater(len(options), 0)

}

// TestDatasetService_Delete 测试数据集删除服务
func (s *DatasetServiceTestSuite) TestDatasetService_Delete() {
	// 创建测试数据集用于删除
	testDataset := &model.SysDataset{
		ParentID:    0,
		Type:        "service_test_type_delete",
		Name:        "待删除的服务测试数据集",
		Description: "待删除的服务测试用数据集",
		Path:        "/service/test/path/delete",
		Status:      1,
		Deleted:     0,
	}
	s.AssertNoError(s.CreateTestData(testDataset))

	// 删除数据集
	ids := []int64{testDataset.ID}
	err := s.datasetService.DeleteDatasets(ids)
	s.AssertNoError(err)

	// 验证数据集已被逻辑删除
	var deletedDataset model.SysDataset
	err = s.GetDB().Where("id = ? AND deleted = ?", testDataset.ID, 0).First(&deletedDataset).Error
	s.AssertError(err) // 应该找不到未删除的记录

}

// 运行测试套件
func TestDatasetService(t *testing.T) {
	suite.Run(t, new(DatasetServiceTestSuite))
}

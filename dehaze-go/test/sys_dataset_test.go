package test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/earthyzinc/dehaze-go/api"
	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func TestDatasetAPI(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

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

	t.Run("TestDatasetCRUD", func(t *testing.T) {
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

		assert.Equal(t, http.StatusOK, resp.Code)

		// 查询数据集列表
		req, _ = http.NewRequest("GET", "/api/v1/dataset", nil)
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		assert.Equal(t, http.StatusOK, resp.Code)

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

		assert.Equal(t, http.StatusOK, resp.Code)
	})

	t.Run("TestDatasetOptions", func(t *testing.T) {
		// 获取数据集下拉选项
		req, _ := http.NewRequest("GET", "/api/v1/dataset/options", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		assert.Equal(t, http.StatusOK, resp.Code)
	})

	t.Run("TestDatasetDelete", func(t *testing.T) {
		// 删除数据集
		req, _ := http.NewRequest("DELETE", "/api/v1/dataset?ids=1", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		assert.Equal(t, http.StatusOK, resp.Code)
	})
}

func TestDatasetService(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	datasetService := service.ServiceGroupApp.DatasetService

	// 清理可能存在的测试数据
	global.DB.Where("name LIKE ?", "%测试%").Delete(&model.SysDataset{})

	t.Run("TestDatasetServiceCRUD", func(t *testing.T) {
		// 创建数据集
		datasetForm := bo.DatasetFormBO{
			ParentID:    0,
			Type:        "service_test_type",
			Name:        "服务测试数据集",
			Description: "服务测试用数据集",
			Path:        "/service/test/path",
			Status:      1,
		}

		err := datasetService.SaveDataset(datasetForm)
		assert.NoError(t, err)

		// 查询数据集列表
		queryParams := struct {
			Keywords string `json:"keywords"`
		}{
			Keywords: "服务测试",
		}

		datasetVOs, err := datasetService.GetDatasetList(queryParams)
		assert.NoError(t, err)
		assert.Greater(t, len(datasetVOs), 0)

		// 获取刚创建的数据集的ID
		var dataset model.SysDataset
		global.DB.Where("name = ?", "服务测试数据集").First(&dataset)
		datasetID := dataset.ID

		// 获取数据集表单
		datasetFormBO, err := datasetService.GetDatasetForm(datasetID)
		assert.NoError(t, err)
		assert.Equal(t, "服务测试数据集", datasetFormBO.Name)

		// 更新数据集
		updateDatasetForm := bo.DatasetFormBO{
			ParentID:    0,
			Type:        "service_test_type_update",
			Name:        "更新后的服务测试数据集",
			Description: "更新后的服务测试用数据集",
			Path:        "/service/test/path/update",
			Status:      1,
		}

		err = datasetService.UpdateDataset(datasetID, updateDatasetForm)
		assert.NoError(t, err)
	})

	t.Run("TestDatasetOptionsService", func(t *testing.T) {
		// 获取数据集下拉选项
		options, err := datasetService.GetDatasetOptions()
		assert.NoError(t, err)
		assert.Greater(t, len(options), 0)
	})

	t.Run("TestDatasetDeleteService", func(t *testing.T) {
		// 获取刚创建的数据集的ID
		var dataset model.SysDataset
		global.DB.Where("name = ?", "更新后的服务测试数据集").First(&dataset)
		datasetID := dataset.ID

		// 删除数据集
		ids := []int64{datasetID}
		err := datasetService.DeleteDatasets(ids)
		assert.NoError(t, err)

		// 验证数据集已被逻辑删除
		var deletedDataset model.SysDataset
		err = global.DB.Where("id = ? AND deleted = ?", datasetID, 0).First(&deletedDataset).Error
		assert.Error(t, err) // 应该找不到未删除的记录
	})
}
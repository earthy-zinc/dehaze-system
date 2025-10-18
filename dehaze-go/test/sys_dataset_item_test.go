package test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func TestSysDatasetItem(t *testing.T) {
	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	// 自动迁移SysDatasetItem表
	err := global.DB.AutoMigrate(&model.SysDatasetItem{})
	if err != nil {
		t.Fatalf("自动迁移SysDatasetItem表失败: %v", err)
	}

	// 创建测试路由
	gin.SetMode(gin.TestMode)
	router := gin.New()
	// initialize.Routers()  // 这里可能需要根据实际情况决定是否调用

	// 创建测试数据集
	dataset := model.SysDataset{
		Name:   "测试数据集",
		Status: 1,
	}
	result := global.DB.Create(&dataset)
	if result.Error != nil {
		t.Fatalf("创建测试数据集失败: %v", result.Error)
	}
	defer global.DB.Delete(&dataset)

	t.Run("CreateDatasetItem", func(t *testing.T) {
		// 创建请求
		req, _ := http.NewRequest("POST", fmt.Sprintf("/api/v1/dataset/item?datasetId=%d", dataset.ID), nil)
		
		// 创建响应记录器
		resp := httptest.NewRecorder()
		
		// 执行请求
		router.ServeHTTP(resp, req)
		
		// 验证响应
		assert.Equal(t, http.StatusOK, resp.Code)
	})

	t.Run("CreateDatasetItemWithName", func(t *testing.T) {
		// 创建请求
		req, _ := http.NewRequest("POST", fmt.Sprintf("/api/v1/dataset/item?datasetId=%d&name=测试数据项", dataset.ID), nil)
		
		// 创建响应记录器
		resp := httptest.NewRecorder()
		
		// 执行请求
		router.ServeHTTP(resp, req)
		
		// 验证响应
		assert.Equal(t, http.StatusOK, resp.Code)
	})

	t.Run("UpdateDatasetItem", func(t *testing.T) {
		// 先创建一个数据项用于更新
		datasetItem := model.SysDatasetItem{
			DatasetID: dataset.ID,
			Name:      "原始名称",
		}
		result := global.DB.Create(&datasetItem)
		assert.NoError(t, result.Error)
		defer global.DB.Delete(&datasetItem)

		// 创建请求
		req, _ := http.NewRequest("PUT", fmt.Sprintf("/api/v1/dataset/item?datasetItemId=%d&name=更新名称", datasetItem.ID), nil)
		
		// 创建响应记录器
		resp := httptest.NewRecorder()
		
		// 执行请求
		router.ServeHTTP(resp, req)
		
		// 验证响应
		assert.Equal(t, http.StatusOK, resp.Code)
	})

	t.Run("DeleteDatasetItem", func(t *testing.T) {
		// 先创建一个数据项用于删除
		datasetItem := model.SysDatasetItem{
			DatasetID: dataset.ID,
			Name:      "待删除项",
		}
		result := global.DB.Create(&datasetItem)
		assert.NoError(t, result.Error)

		// 创建请求
		req, _ := http.NewRequest("DELETE", fmt.Sprintf("/api/v1/dataset/item?datasetItemId=%d", datasetItem.ID), nil)
		
		// 创建响应记录器
		resp := httptest.NewRecorder()
		
		// 执行请求
		router.ServeHTTP(resp, req)
		
		// 验证响应
		assert.Equal(t, http.StatusOK, resp.Code)
	})
}
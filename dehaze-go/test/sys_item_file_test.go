package test

import (
	"bytes"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func TestSysItemFile(t *testing.T) {
	// 初始化配置和数据库
	initialize.Viper()
	initialize.Gorm()

	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	// 自动迁移相关表
	err := global.DB.AutoMigrate(&model.SysDataset{}, &model.SysDatasetItem{}, &model.SysItemFile{}, &model.SysFile{})
	if err != nil {
		t.Fatalf("自动迁移表失败: %v", err)
	}

	// 创建测试路由
	gin.SetMode(gin.TestMode)
	router := gin.New()
	initialize.Routers()

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

	// 创建测试数据项
	datasetItem := model.SysDatasetItem{
		DatasetID: dataset.ID,
		Name:      "测试数据项",
	}
	result = global.DB.Create(&datasetItem)
	if result.Error != nil {
		t.Fatalf("创建测试数据项失败: %v", result.Error)
	}
	defer global.DB.Delete(&datasetItem)

	t.Run("AddImageById", func(t *testing.T) {
		// 准备multipart表单数据
		var b bytes.Buffer
		w := multipart.NewWriter(&b)
		
		// 添加文件字段
		fw, err := w.CreateFormFile("file", "test.txt")
		assert.NoError(t, err)
		
		// 写入文件内容
		_, err = io.WriteString(fw, "This is a test file")
		assert.NoError(t, err)
		
		// 添加其他字段
		w.WriteField("datasetId", fmt.Sprintf("%d", dataset.ID))
		w.WriteField("datasetItemId", fmt.Sprintf("%d", datasetItem.ID))
		w.WriteField("type", "test")
		w.WriteField("description", "测试文件")
		w.Close()

		// 创建请求
		req, _ := http.NewRequest("POST", "/api/v1/dataset/image", &b)
		req.Header.Set("Content-Type", w.FormDataContentType())
		
		// 创建响应记录器
		resp := httptest.NewRecorder()
		
		// 执行请求
		router.ServeHTTP(resp, req)
		
		// 验证响应
		assert.Equal(t, http.StatusOK, resp.Code)
	})

	t.Run("RemoveImageById", func(t *testing.T) {
		// 先创建一个文件记录用于删除
		file := model.SysFile{
			Type:       ".txt",
			URL:        "http://localhost/test.txt",
			Name:       "test.txt",
			ObjectName: "test/test.txt",
			Size:       "18",
			Path:       "/tmp/test.txt",
			MD5:        "test_md5_delete",
		}
		result := global.DB.Create(&file)
		assert.NoError(t, result.Error)
		defer global.DB.Delete(&file)

		// 创建项文件关联记录
		itemFile := model.SysItemFile{
			ItemID:      datasetItem.ID,
			FileID:      file.ID,
			Type:        "test",
			Description: "测试文件",
		}
		result = global.DB.Create(&itemFile)
		assert.NoError(t, result.Error)
		defer global.DB.Delete(&itemFile)

		// 创建请求
		req, _ := http.NewRequest("DELETE", fmt.Sprintf("/api/v1/dataset/image?itemFileId=%d", itemFile.ID), nil)
		
		// 创建响应记录器
		resp := httptest.NewRecorder()
		
		// 执行请求
		router.ServeHTTP(resp, req)
		
		// 验证响应
		assert.Equal(t, http.StatusOK, resp.Code)
	})
}

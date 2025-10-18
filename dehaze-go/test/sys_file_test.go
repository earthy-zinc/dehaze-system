package test

import (
	"bytes"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func TestSysFile(t *testing.T) {
	// 初始化配置和数据库
	initialize.Viper()
	initialize.Gorm()

	// 检查数据库连接是否可用
	if global.DB == nil {
		t.Skip("数据库连接不可用，跳过测试")
	}

	// 自动迁移SysFile表
	err := global.DB.AutoMigrate(&model.SysFile{})
	if err != nil {
		t.Fatalf("自动迁移SysFile表失败: %v", err)
	}

	// 创建测试路由
	gin.SetMode(gin.TestMode)
	router := gin.New()
	initialize.Routers()

	// 创建临时测试文件
	tempFile, err := createTempFile()
	if err != nil {
		t.Fatalf("创建临时文件失败: %v", err)
	}
	defer os.Remove(tempFile.Name())

	t.Run("UploadFile", func(t *testing.T) {
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
		w.WriteField("modelId", "1")
		w.Close()

		// 创建请求
		req, _ := http.NewRequest("POST", "/api/v1/files", &b)
		req.Header.Set("Content-Type", w.FormDataContentType())
		
		// 创建响应记录器
		resp := httptest.NewRecorder()
		
		// 执行请求
		router.ServeHTTP(resp, req)
		
		// 验证响应
		assert.Equal(t, http.StatusOK, resp.Code)
	})

	t.Run("CheckFile", func(t *testing.T) {
		// 创建请求
		req, _ := http.NewRequest("GET", "/api/v1/files/check?md5=test_md5", nil)
		
		// 创建响应记录器
		resp := httptest.NewRecorder()
		
		// 执行请求
		router.ServeHTTP(resp, req)
		
		// 验证响应
		assert.Equal(t, http.StatusOK, resp.Code)
	})

	t.Run("DeleteFile", func(t *testing.T) {
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

		// 创建请求
		req, _ := http.NewRequest("DELETE", fmt.Sprintf("/api/v1/files?fileId=%d", file.ID), nil)
		
		// 创建响应记录器
		resp := httptest.NewRecorder()
		
		// 执行请求
		router.ServeHTTP(resp, req)
		
		// 验证响应
		assert.Equal(t, http.StatusOK, resp.Code)
	})
}

// createTempFile 创建临时文件用于测试
func createTempFile() (*os.File, error) {
	// 创建临时文件
	tempFile, err := os.CreateTemp("", "test_*.txt")
	if err != nil {
		return nil, err
	}

	// 写入测试内容
	content := "This is a test file for upload testing"
	_, err = tempFile.WriteString(content)
	if err != nil {
		tempFile.Close()
		return nil, err
	}

	// 重置文件指针
	_, err = tempFile.Seek(0, 0)
	if err != nil {
		tempFile.Close()
		return nil, err
	}

	return tempFile, nil
}
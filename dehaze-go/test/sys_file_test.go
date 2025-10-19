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

	"github.com/earthyzinc/dehaze-go/initialize"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/utils"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/suite"
)

// FileTestSuite 文件服务测试套件
// 使用事务隔离，每个测试方法都在独立事务中运行
type FileTestSuite struct {
	BaseTestSuite
}

// SetupSuite 在整个测试套件开始前运行一次
func (s *FileTestSuite) SetupSuite() {
}

// TestUploadFile 测试文件上传
func (s *FileTestSuite) TestUploadFile() {
	// 自动迁移SysFile表
	err := s.GetDB().AutoMigrate(&model.SysFile{})
	s.Require().NoError(err, "自动迁移SysFile表失败")

	// 创建测试路由
	gin.SetMode(gin.TestMode)
	router := gin.New()
	initialize.Routers()

	// 创建临时测试文件
	tempFile, err := s.createTempFile()
	s.Require().NoError(err, "创建临时文件失败")
	defer os.Remove(tempFile.Name())

	// 准备multipart表单数据
	var b bytes.Buffer
	w := multipart.NewWriter(&b)

	// 添加文件字段
	fw, err := w.CreateFormFile("file", "test.txt")
	s.Require().NoError(err)

	// 写入文件内容
	_, err = io.WriteString(fw, "This is a test file")
	s.Require().NoError(err)

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
	s.Assert().Equal(http.StatusOK, resp.Code)

}

// TestCheckFile 测试文件检查
func (s *FileTestSuite) TestCheckFile() {
	// 自动迁移SysFile表
	err := s.GetDB().AutoMigrate(&model.SysFile{})
	s.Require().NoError(err, "自动迁移SysFile表失败")

	// 创建测试路由
	gin.SetMode(gin.TestMode)
	router := gin.New()
	initialize.Routers()

	// 创建请求
	req, _ := http.NewRequest("GET", "/api/v1/files/check?md5=test_md5", nil)

	// 创建响应记录器
	resp := httptest.NewRecorder()

	// 执行请求
	router.ServeHTTP(resp, req)

	// 验证响应
	s.Assert().Equal(http.StatusOK, resp.Code)

}

// TestDeleteFile 测试文件删除
func (s *FileTestSuite) TestDeleteFile() {
	// 自动迁移SysFile表
	err := s.GetDB().AutoMigrate(&model.SysFile{})
	s.Require().NoError(err, "自动迁移SysFile表失败")

	// 创建测试路由
	gin.SetMode(gin.TestMode)
	router := gin.New()
	initialize.Routers()

	// 先创建一个文件记录用于删除
	file := &model.SysFile{
		Type:       utils.StringPtr(".txt"),
		URL:        utils.StringPtr("http://localhost/test.txt"),
		Name:       "test.txt",
		ObjectName: "test/test.txt",
		Size:       "18",
		Path:       "/tmp/test.txt",
		MD5:        "test_md5_delete",
	}
	s.Require().NoError(s.CreateTestData(file))

	// 创建请求
	req, _ := http.NewRequest("DELETE", fmt.Sprintf("/api/v1/files?fileId=%d", file.ID), nil)

	// 创建响应记录器
	resp := httptest.NewRecorder()

	// 执行请求
	router.ServeHTTP(resp, req)

	// 验证响应
	s.Assert().Equal(http.StatusOK, resp.Code)

}

// createTempFile 创建临时文件用于测试
func (s *FileTestSuite) createTempFile() (*os.File, error) {
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

// 运行测试套件
func TestFileService(t *testing.T) {
	suite.Run(t, new(FileTestSuite))
}

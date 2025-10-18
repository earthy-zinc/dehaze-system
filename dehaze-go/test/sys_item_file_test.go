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
	"github.com/stretchr/testify/suite"
)

// ItemFileTestSuite 项文件测试套件
// 使用事务隔离，每个测试方法都在独立事务中运行
type ItemFileTestSuite struct {
	TransactionTestSuite
}

// SetupSuite 在整个测试套件开始前运行一次
func (s *ItemFileTestSuite) SetupSuite() {
	// 初始化配置和数据库
	initialize.Viper()
	initialize.Gorm()

	// 检查数据库连接是否可用
	if global.DB == nil {
		s.T().Skip("数据库连接不可用，跳过测试")
	}
}

// TestAddImageById 测试通过ID添加图片
func (s *ItemFileTestSuite) TestAddImageById() {
	// 自动迁移相关表
	err := s.GetDB().AutoMigrate(&model.SysDataset{}, &model.SysDatasetItem{}, &model.SysItemFile{}, &model.SysFile{})
	s.Require().NoError(err, "自动迁移表失败")

	// 创建测试路由
	gin.SetMode(gin.TestMode)
	router := gin.New()
	initialize.Routers()

	// 创建测试数据集
	dataset := &model.SysDataset{
		Name:   "测试数据集",
		Status: 1,
	}
	s.Require().NoError(s.CreateTestData(dataset))

	// 创建测试数据项
	datasetItem := &model.SysDatasetItem{
		DatasetID: dataset.ID,
		Name:      "测试数据项",
	}
	s.Require().NoError(s.CreateTestData(datasetItem))

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
	s.Assert().Equal(http.StatusOK, resp.Code)

}

// TestRemoveImageById 测试通过ID移除图片
func (s *ItemFileTestSuite) TestRemoveImageById() {
	// 自动迁移相关表
	err := s.GetDB().AutoMigrate(&model.SysDataset{}, &model.SysDatasetItem{}, &model.SysItemFile{}, &model.SysFile{})
	s.Require().NoError(err, "自动迁移表失败")

	// 创建测试路由
	gin.SetMode(gin.TestMode)
	router := gin.New()
	initialize.Routers()

	// 创建测试数据集
	dataset := &model.SysDataset{
		Name:   "测试数据集",
		Status: 1,
	}
	s.Require().NoError(s.CreateTestData(dataset))

	// 创建测试数据项
	datasetItem := &model.SysDatasetItem{
		DatasetID: dataset.ID,
		Name:      "测试数据项",
	}
	s.Require().NoError(s.CreateTestData(datasetItem))

	// 先创建一个文件记录用于删除
	file := &model.SysFile{
		Type:       ".txt",
		URL:        "http://localhost/test.txt",
		Name:       "test.txt",
		ObjectName: "test/test.txt",
		Size:       "18",
		Path:       "/tmp/test.txt",
		MD5:        "test_md5_delete",
	}
	s.Require().NoError(s.CreateTestData(file))

	// 创建项文件关联记录
	itemFile := &model.SysItemFile{
		ItemID:      datasetItem.ID,
		FileID:      file.ID,
		Type:        "test",
		Description: "测试文件",
	}
	s.Require().NoError(s.CreateTestData(itemFile))

	// 创建请求
	req, _ := http.NewRequest("DELETE", fmt.Sprintf("/api/v1/dataset/image?itemFileId=%d", itemFile.ID), nil)

	// 创建响应记录器
	resp := httptest.NewRecorder()

	// 执行请求
	router.ServeHTTP(resp, req)

	// 验证响应
	s.Assert().Equal(http.StatusOK, resp.Code)

}

// 运行测试套件
func TestItemFileService(t *testing.T) {
	suite.Run(t, new(ItemFileTestSuite))
}

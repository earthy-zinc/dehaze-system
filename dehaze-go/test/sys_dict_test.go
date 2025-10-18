package test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/earthyzinc/dehaze-go/api"
	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/initialize"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/model/vo"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/suite"
)

// DictServiceTestSuite 字典服务测试套件
// 使用事务隔离，每个测试方法都在独立事务中运行
type DictServiceTestSuite struct {
	TransactionTestSuite
	dictService     *service.DictService
	dictTypeService *service.DictTypeService
}

// SetupSuite 在整个测试套件开始前运行一次
func (s *DictServiceTestSuite) SetupSuite() {
	// 初始化zap日志
	initialize.Zap()
	initialize.LocalCache()

	// 初始化服务
	s.dictService = &service.ServiceGroupApp.DictService
	s.dictTypeService = &service.ServiceGroupApp.DictTypeService
}

// TestDictAPI_DictTypeCRUD 测试字典类型API的增删改查功能
func (s *DictServiceTestSuite) TestDictAPI_DictTypeCRUD() {
	// 创建测试路由器
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(gin.Recovery())

	// 注册路由组
	apiGroup := router.Group("/api/v1")
	{
		apiGroup.GET("/dict/page", api.ApiGroupApp.SysDictApi.GetDictPage)
		apiGroup.GET("/dict/:id/form", api.ApiGroupApp.SysDictApi.GetDictForm)
		apiGroup.POST("/dict", api.ApiGroupApp.SysDictApi.SaveDict)
		apiGroup.PUT("/dict/:id", api.ApiGroupApp.SysDictApi.UpdateDict)
		apiGroup.DELETE("/dict/:ids", api.ApiGroupApp.SysDictApi.DeleteDict)
		apiGroup.GET("/dict/:typeCode/options", api.ApiGroupApp.SysDictApi.ListDictOptions)

		apiGroup.GET("/dict/types/page", api.ApiGroupApp.SysDictApi.GetDictTypePage)
		apiGroup.GET("/dict/types/:id/form", api.ApiGroupApp.SysDictApi.GetDictTypeForm)
		apiGroup.POST("/dict/types", api.ApiGroupApp.SysDictApi.SaveDictType)
		apiGroup.PUT("/dict/types/:id", api.ApiGroupApp.SysDictApi.UpdateDictType)
		apiGroup.DELETE("/dict/types/:ids", api.ApiGroupApp.SysDictApi.DeleteDictTypes)
	}

	// 创建字典类型
	dictTypeForm := bo.DictTypeFormBO{
		Name:   "测试字典类型",
		Code:   "TEST_TYPE",
		Status: 1,
		Remark: "测试用字典类型",
	}

	jsonValue, _ := json.Marshal(dictTypeForm)
	req, _ := http.NewRequest("POST", "/api/v1/dict/types", bytes.NewBuffer(jsonValue))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	s.Assert().Equal(http.StatusOK, resp.Code)

	// 查询字典类型列表
	req, _ = http.NewRequest("GET", "/api/v1/dict/types/page", nil)
	resp = httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	s.Assert().Equal(http.StatusOK, resp.Code)

	// 获取字典类型表单数据
	req, _ = http.NewRequest("GET", "/api/v1/dict/types/1/form", nil)
	resp = httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	// 更新字典类型
	updateDictTypeForm := bo.DictTypeFormBO{
		Name:   "更新后的字典类型",
		Code:   "TEST_TYPE_UPDATE",
		Status: 1,
		Remark: "更新后的测试用字典类型",
	}

	jsonValue, _ = json.Marshal(updateDictTypeForm)
	req, _ = http.NewRequest("PUT", "/api/v1/dict/types/1", bytes.NewBuffer(jsonValue))
	req.Header.Set("Content-Type", "application/json")
	resp = httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	s.Assert().Equal(http.StatusOK, resp.Code)

}

// TestDictAPI_DictCRUD 测试字典项API的增删改查功能
func (s *DictServiceTestSuite) TestDictAPI_DictCRUD() {
	// 创建测试路由器
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(gin.Recovery())

	// 注册路由组
	apiGroup := router.Group("/api/v1")
	{
		apiGroup.GET("/dict/page", api.ApiGroupApp.SysDictApi.GetDictPage)
		apiGroup.GET("/dict/:id/form", api.ApiGroupApp.SysDictApi.GetDictForm)
		apiGroup.POST("/dict", api.ApiGroupApp.SysDictApi.SaveDict)
		apiGroup.PUT("/dict/:id", api.ApiGroupApp.SysDictApi.UpdateDict)
		apiGroup.DELETE("/dict/:ids", api.ApiGroupApp.SysDictApi.DeleteDict)
		apiGroup.GET("/dict/:typeCode/options", api.ApiGroupApp.SysDictApi.ListDictOptions)

		apiGroup.GET("/dict/types/page", api.ApiGroupApp.SysDictApi.GetDictTypePage)
		apiGroup.GET("/dict/types/:id/form", api.ApiGroupApp.SysDictApi.GetDictTypeForm)
		apiGroup.POST("/dict/types", api.ApiGroupApp.SysDictApi.SaveDictType)
		apiGroup.PUT("/dict/types/:id", api.ApiGroupApp.SysDictApi.UpdateDictType)
		apiGroup.DELETE("/dict/types/:ids", api.ApiGroupApp.SysDictApi.DeleteDictTypes)
	}

	// 创建字典项
	dictForm := bo.DictFormBO{
		TypeCode: "TEST_TYPE_UPDATE",
		Name:     "测试字典项",
		Value:    "TEST_VALUE",
		Status:   1,
		Sort:     1,
		Remark:   "测试用字典项",
	}

	jsonValue, _ := json.Marshal(dictForm)
	req, _ := http.NewRequest("POST", "/api/v1/dict", bytes.NewBuffer(jsonValue))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	s.Assert().Equal(http.StatusOK, resp.Code)

	// 查询字典项列表
	req, _ = http.NewRequest("GET", "/api/v1/dict/page", nil)
	resp = httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	s.Assert().Equal(http.StatusOK, resp.Code)

	// 获取字典项表单数据
	req, _ = http.NewRequest("GET", "/api/v1/dict/1/form", nil)
	resp = httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	// 更新字典项
	updateDictForm := bo.DictFormBO{
		TypeCode: "TEST_TYPE_UPDATE",
		Name:     "更新后的字典项",
		Value:    "TEST_VALUE_UPDATE",
		Status:   1,
		Sort:     2,
		Remark:   "更新后的测试用字典项",
	}

	jsonValue, _ = json.Marshal(updateDictForm)
	req, _ = http.NewRequest("PUT", "/api/v1/dict/1", bytes.NewBuffer(jsonValue))
	req.Header.Set("Content-Type", "application/json")
	resp = httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	s.Assert().Equal(http.StatusOK, resp.Code)

	// 获取字典下拉选项
	req, _ = http.NewRequest("GET", "/api/v1/dict/TEST_TYPE_UPDATE/options", nil)
	resp = httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	s.Assert().Equal(http.StatusOK, resp.Code)

}

// TestDictAPI_Delete 测试字典删除API
func (s *DictServiceTestSuite) TestDictAPI_Delete() {
	// 创建测试路由器
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(gin.Recovery())

	// 注册路由组
	apiGroup := router.Group("/api/v1")
	{
		apiGroup.GET("/dict/page", api.ApiGroupApp.SysDictApi.GetDictPage)
		apiGroup.GET("/dict/:id/form", api.ApiGroupApp.SysDictApi.GetDictForm)
		apiGroup.POST("/dict", api.ApiGroupApp.SysDictApi.SaveDict)
		apiGroup.PUT("/dict/:id", api.ApiGroupApp.SysDictApi.UpdateDict)
		apiGroup.DELETE("/dict/:ids", api.ApiGroupApp.SysDictApi.DeleteDict)
		apiGroup.GET("/dict/:typeCode/options", api.ApiGroupApp.SysDictApi.ListDictOptions)

		apiGroup.GET("/dict/types/page", api.ApiGroupApp.SysDictApi.GetDictTypePage)
		apiGroup.GET("/dict/types/:id/form", api.ApiGroupApp.SysDictApi.GetDictTypeForm)
		apiGroup.POST("/dict/types", api.ApiGroupApp.SysDictApi.SaveDictType)
		apiGroup.PUT("/dict/types/:id", api.ApiGroupApp.SysDictApi.UpdateDictType)
		apiGroup.DELETE("/dict/types/:ids", api.ApiGroupApp.SysDictApi.DeleteDictTypes)
	}

	// 删除字典项
	req, _ := http.NewRequest("DELETE", "/api/v1/dict/1", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	s.Assert().Equal(http.StatusOK, resp.Code)

	// 删除字典类型
	req, _ = http.NewRequest("DELETE", "/api/v1/dict/types/1", nil)
	resp = httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	s.Assert().Equal(http.StatusOK, resp.Code)

}

// TestDictService_DictType 测试字典类型服务功能
func (s *DictServiceTestSuite) TestDictService_DictType() {
	// 创建字典类型
	dictTypeForm := bo.DictTypeFormBO{
		Name:   "服务测试字典类型",
		Code:   "SERVICE_TEST_TYPE",
		Status: 1,
		Remark: "服务测试用字典类型",
	}

	err := s.dictTypeService.SaveDictType(dictTypeForm)
	s.AssertNoError(err)

	// 查询字典类型分页
	queryParams := struct {
		Keywords string `json:"keywords"`
		PageNum  int    `json:"pageNum"`
		PageSize int    `json:"pageSize"`
	}{
		Keywords: "服务测试",
		PageNum:  1,
		PageSize: 10,
	}

	result, err := s.dictTypeService.GetDictTypePage(queryParams)
	s.AssertNoError(err)
	s.Assert().Greater(result.Total, int64(0))

	// 获取字典类型表单
	dictTypeFormBO, err := s.dictTypeService.GetDictTypeForm(1)
	s.AssertNoError(err)
	s.AssertEqual("服务测试字典类型", dictTypeFormBO.Name)

	// 更新字典类型
	updateDictTypeForm := bo.DictTypeFormBO{
		Name:   "更新后的服务测试字典类型",
		Code:   "SERVICE_TEST_TYPE_UPDATE",
		Status: 1,
		Remark: "更新后的服务测试用字典类型",
	}

	err = s.dictTypeService.UpdateDictType(1, updateDictTypeForm)
	s.AssertNoError(err)

}

// TestDictService_Dict 测试字典项服务功能
func (s *DictServiceTestSuite) TestDictService_Dict() {
	// 创建字典项
	dictForm := bo.DictFormBO{
		TypeCode: "SERVICE_TEST_TYPE_UPDATE",
		Name:     "服务测试字典项",
		Value:    "SERVICE_TEST_VALUE",
		Status:   1,
		Sort:     1,
		Remark:   "服务测试用字典项",
	}

	err := s.dictService.SaveDict(dictForm)
	s.AssertNoError(err)

	// 查询字典项分页
	queryParams := struct {
		Keywords string `json:"keywords"`
		TypeCode string `json:"typeCode"`
		PageNum  int    `json:"pageNum"`
		PageSize int    `json:"pageSize"`
	}{
		Keywords: "服务测试",
		TypeCode: "SERVICE_TEST_TYPE_UPDATE",
		PageNum:  1,
		PageSize: 10,
	}

	result, err := s.dictService.GetDictPage(queryParams)
	s.AssertNoError(err)
	s.Assert().Greater(result.Total, int64(0))

	// 获取字典项表单
	dictFormBO, err := s.dictService.GetDictForm(1)
	s.AssertNoError(err)
	s.AssertEqual("服务测试字典项", dictFormBO.Name)

	// 更新字典项
	updateDictForm := bo.DictFormBO{
		TypeCode: "SERVICE_TEST_TYPE_UPDATE",
		Name:     "更新后的服务测试字典项",
		Value:    "SERVICE_TEST_VALUE_UPDATE",
		Status:   1,
		Sort:     2,
		Remark:   "更新后的服务测试用字典项",
	}

	err = s.dictService.UpdateDict(1, updateDictForm)
	s.AssertNoError(err)

	// 获取字典下拉选项
	options, err := s.dictService.ListDictOptions("SERVICE_TEST_TYPE_UPDATE")
	s.AssertNoError(err)
	s.Assert().Greater(len(options), 0)

}

// TestDictService_Delete 测试字典删除服务
func (s *DictServiceTestSuite) TestDictService_Delete() {
	// 删除字典项
	err := s.dictService.DeleteDict("1")
	s.AssertNoError(err)

	// 删除字典类型
	err = s.dictTypeService.DeleteDictTypes("1")
	s.AssertNoError(err)

}

// TestGetDictPage_DBError 测试数据库错误情况
func (s *DictServiceTestSuite) TestGetDictPage_DBError() {
	// 模拟数据库连接断开的情况
	originalDB := s.DB
	s.DB = nil
	global.DB = nil

	// 执行查询
	queryParams := query.DictPageQuery{
		PageNum:  1,
		PageSize: 10,
	}
	pageResult, err := s.dictService.GetDictPage(queryParams)

	// 恢复原始数据库连接
	s.DB = originalDB
	global.DB = originalDB

	// 验证结果
	s.AssertError(err)
	s.AssertEqual(vo.PageResult[vo.DictPageVO]{}, pageResult)
}

// TestGetDictForm_DBError 测试数据库错误情况
func (s *DictServiceTestSuite) TestGetDictForm_DBError() {
	// 模拟数据库连接断开的情况
	originalDB := s.DB
	s.DB = nil
	global.DB = nil

	// 获取表单数据
	dictFormBO, err := s.dictService.GetDictForm(1)

	// 恢复原始数据库连接
	s.DB = originalDB
	global.DB = originalDB

	// 验证结果
	s.AssertError(err)
	s.AssertEqual(bo.DictFormBO{}, dictFormBO)
}

// TestSaveDict_DBError 测试保存字典时数据库错误
func (s *DictServiceTestSuite) TestSaveDict_DBError() {
	// 准备字典表单数据
	dictFormBO := bo.DictFormBO{
		TypeCode: "test_type_db_error",
		Name:     "测试数据库错误字典",
		Value:    "test_value_db_error",
		Status:   1,
		Sort:     1,
		Remark:   "测试数据库错误字典备注",
	}

	// 模拟数据库连接断开的情况
	originalDB := s.DB
	s.DB = nil
	global.DB = nil

	// 保存字典
	err := s.dictService.SaveDict(dictFormBO)

	// 恢复原始数据库连接
	s.DB = originalDB
	global.DB = originalDB

	// 验证结果
	s.AssertError(err)
}

// TestUpdateDict_DBError 测试更新字典时数据库错误
func (s *DictServiceTestSuite) TestUpdateDict_DBError() {
	// 准备更新数据
	dictFormBO := bo.DictFormBO{
		TypeCode: "test_type_update_db_error",
		Name:     "测试更新数据库错误字典",
		Value:    "test_value_update_db_error",
		Status:   1,
		Sort:     1,
		Remark:   "测试更新数据库错误字典备注",
	}

	// 模拟数据库连接断开的情况
	originalDB := s.DB
	s.DB = nil
	global.DB = nil

	// 更新字典
	err := s.dictService.UpdateDict(1, dictFormBO)

	// 恢复原始数据库连接
	s.DB = originalDB
	global.DB = originalDB

	// 验证结果
	s.AssertError(err)
}

// TestDeleteDict_DBError 测试删除字典时数据库错误
func (s *DictServiceTestSuite) TestDeleteDict_DBError() {
	// 模拟数据库连接断开的情况
	originalDB := s.DB
	s.DB = nil
	global.DB = nil

	// 删除字典
	err := s.dictService.DeleteDict("1,2,3")

	// 恢复原始数据库连接
	s.DB = originalDB
	global.DB = originalDB

	// 验证结果
	s.AssertError(err)
}

// TestListDictOptions_DBError 测试获取字典下拉列表时数据库错误
func (s *DictServiceTestSuite) TestListDictOptions_DBError() {
	// 模拟数据库连接断开的情况
	originalDB := s.DB
	s.DB = nil
	global.DB = nil

	// 获取字典下拉列表
	options, err := s.dictService.ListDictOptions("test_type")

	// 恢复原始数据库连接
	s.DB = originalDB
	global.DB = originalDB

	// 验证结果
	s.AssertError(err)
	s.Assert().Nil(options)
}

// 运行测试套件
func TestDictServiceSuite(t *testing.T) {
	suite.Run(t, new(DictServiceTestSuite))
}

package test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/earthyzinc/dehaze-go/api"
	"github.com/earthyzinc/dehaze-go/initialize"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func setupDictTest() {
	// 初始化zap日志
	initialize.Zap()
	initialize.LocalCache()
	//initialize.Gorm()
}

func TestDictAPI(t *testing.T) {
	setupDictTest()

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

	t.Run("TestDictTypeCRUD", func(t *testing.T) {
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

		assert.Equal(t, http.StatusOK, resp.Code)

		// 查询字典类型列表
		req, _ = http.NewRequest("GET", "/api/v1/dict/types/page", nil)
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		assert.Equal(t, http.StatusOK, resp.Code)

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

		assert.Equal(t, http.StatusOK, resp.Code)
	})

	t.Run("TestDictCRUD", func(t *testing.T) {
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

		assert.Equal(t, http.StatusOK, resp.Code)

		// 查询字典项列表
		req, _ = http.NewRequest("GET", "/api/v1/dict/page", nil)
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		assert.Equal(t, http.StatusOK, resp.Code)

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

		assert.Equal(t, http.StatusOK, resp.Code)

		// 获取字典下拉选项
		req, _ = http.NewRequest("GET", "/api/v1/dict/TEST_TYPE_UPDATE/options", nil)
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		assert.Equal(t, http.StatusOK, resp.Code)
	})

	t.Run("TestDictDelete", func(t *testing.T) {
		// 删除字典项
		req, _ := http.NewRequest("DELETE", "/api/v1/dict/1", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		assert.Equal(t, http.StatusOK, resp.Code)

		// 删除字典类型
		req, _ = http.NewRequest("DELETE", "/api/v1/dict/types/1", nil)
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		assert.Equal(t, http.StatusOK, resp.Code)
	})
}

func TestDictService(t *testing.T) {
	setupDictTest()

	dictService := service.ServiceGroupApp.DictService
	dictTypeService := service.ServiceGroupApp.DictTypeService

	t.Run("TestDictTypeService", func(t *testing.T) {
		// 创建字典类型
		dictTypeForm := bo.DictTypeFormBO{
			Name:   "服务测试字典类型",
			Code:   "SERVICE_TEST_TYPE",
			Status: 1,
			Remark: "服务测试用字典类型",
		}

		err := dictTypeService.SaveDictType(dictTypeForm)
		assert.NoError(t, err)

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

		result, err := dictTypeService.GetDictTypePage(queryParams)
		assert.NoError(t, err)
		assert.Greater(t, result.Total, int64(0))

		// 获取字典类型表单
		dictTypeFormBO, err := dictTypeService.GetDictTypeForm(1)
		assert.NoError(t, err)
		assert.Equal(t, "服务测试字典类型", dictTypeFormBO.Name)

		// 更新字典类型
		updateDictTypeForm := bo.DictTypeFormBO{
			Name:   "更新后的服务测试字典类型",
			Code:   "SERVICE_TEST_TYPE_UPDATE",
			Status: 1,
			Remark: "更新后的服务测试用字典类型",
		}

		err = dictTypeService.UpdateDictType(1, updateDictTypeForm)
		assert.NoError(t, err)
	})

	t.Run("TestDictService", func(t *testing.T) {
		// 创建字典项
		dictForm := bo.DictFormBO{
			TypeCode: "SERVICE_TEST_TYPE_UPDATE",
			Name:     "服务测试字典项",
			Value:    "SERVICE_TEST_VALUE",
			Status:   1,
			Sort:     1,
			Remark:   "服务测试用字典项",
		}

		dictService := service.ServiceGroupApp.DictService
		err := dictService.SaveDict(dictForm)
		assert.NoError(t, err)

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

		result, err := dictService.GetDictPage(queryParams)
		assert.NoError(t, err)
		assert.Greater(t, result.Total, int64(0))

		// 获取字典项表单
		dictFormBO, err := dictService.GetDictForm(1)
		assert.NoError(t, err)
		assert.Equal(t, "服务测试字典项", dictFormBO.Name)

		// 更新字典项
		updateDictForm := bo.DictFormBO{
			TypeCode: "SERVICE_TEST_TYPE_UPDATE",
			Name:     "更新后的服务测试字典项",
			Value:    "SERVICE_TEST_VALUE_UPDATE",
			Status:   1,
			Sort:     2,
			Remark:   "更新后的服务测试用字典项",
		}

		err = dictService.UpdateDict(1, updateDictForm)
		assert.NoError(t, err)

		// 获取字典下拉选项
		options, err := dictService.ListDictOptions("SERVICE_TEST_TYPE_UPDATE")
		assert.NoError(t, err)
		assert.Greater(t, len(options), 0)
	})

	t.Run("TestDictDeleteService", func(t *testing.T) {
		// 删除字典项
		err := dictService.DeleteDict("1")
		assert.NoError(t, err)

		// 删除字典类型
		err = dictTypeService.DeleteDictTypes("1")
		assert.NoError(t, err)
	})
}

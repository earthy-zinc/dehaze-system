package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/common"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/gin-gonic/gin"
)

type SysDictApi struct {
	dictService     service.DictService
	dictTypeService service.DictTypeService
}

// GetDictPage 字典分页列表
// @Summary 字典分页列表
// @Description 获取字典分页列表
// @Tags 字典接口
// @Accept application/json
// @Produce application/json
// @Param keywords query string false "关键字(字典项名称)"
// @Param typeCode query string false "字典类型编码"
// @Param pageNum query int false "页码"
// @Param pageSize query int false "每页条数"
// @Success 200 {object} vo.Result{data=vo.PageResult[vo.DictPageVO]}
// @Router /api/v1/dict/page [get]
func (api *SysDictApi) GetDictPage(c *gin.Context) {
	// 解析查询参数
	var queryParams query.DictPageQuery
	queryParams.Keywords = c.Query("keywords")
	queryParams.TypeCode = c.Query("typeCode")

	if pageNumStr := c.Query("pageNum"); pageNumStr != "" {
		if pageNum, err := strconv.Atoi(pageNumStr); err == nil {
			queryParams.PageNum = pageNum
		} else {
			queryParams.PageNum = 1
		}
	} else {
		queryParams.PageNum = 1
	}

	if pageSizeStr := c.Query("pageSize"); pageSizeStr != "" {
		if pageSize, err := strconv.Atoi(pageSizeStr); err == nil {
			queryParams.PageSize = pageSize
		} else {
			queryParams.PageSize = 10
		}
	} else {
		queryParams.PageSize = 10
	}

	// 调用服务获取分页数据
	result, err := api.dictService.GetDictPage(queryParams)
	if err != nil {
		common.FailWithMessage("获取字典分页列表失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(result, "查询成功", c)
}

// GetDictForm 字典数据表单数据
// @Summary 字典数据表单数据
// @Description 获取字典数据表单数据
// @Tags 字典接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "字典ID"
// @Success 200 {object} vo.Result{data=bo.DictFormBO}
// @Router /api/v1/dict/{id}/form [get]
func (api *SysDictApi) GetDictForm(c *gin.Context) {
	// 获取路径参数
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		common.FailWithMessage("字典ID格式不正确", c)
		return
	}

	// 调用服务获取字典表单数据
	dictFormBO, err := api.dictService.GetDictForm(id)
	if err != nil {
		common.FailWithMessage("获取字典表单数据失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(dictFormBO, "查询成功", c)
}

// SaveDict 新增字典
// @Summary 新增字典
// @Description 新增字典
// @Tags 字典接口
// @Accept application/json
// @Produce application/json
// @Param DictForm body bo.DictFormBO true "字典表单数据"
// @Success 200 {object} vo.Result
// @Router /api/v1/dict [post]
func (api *SysDictApi) SaveDict(c *gin.Context) {
	// 绑定请求参数
	var dictFormBO bo.DictFormBO
	if err := c.ShouldBindJSON(&dictFormBO); err != nil {
		common.FailWithMessage("请求参数解析失败: "+err.Error(), c)
		return
	}

	// 调用服务保存字典
	err := api.dictService.SaveDict(dictFormBO)
	if err != nil {
		common.FailWithMessage("新增字典失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("新增字典成功", c)
}

// UpdateDict 修改字典
// @Summary 修改字典
// @Description 修改字典
// @Tags 字典接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "字典ID"
// @Param DictForm body bo.DictFormBO true "字典表单数据"
// @Success 200 {object} vo.Result
// @Router /api/v1/dict/{id} [put]
func (api *SysDictApi) UpdateDict(c *gin.Context) {
	// 获取路径参数
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		common.FailWithMessage("字典ID格式不正确", c)
		return
	}

	// 绑定请求参数
	var dictFormBO bo.DictFormBO
	if err := c.ShouldBindJSON(&dictFormBO); err != nil {
		common.FailWithMessage("请求参数解析失败: "+err.Error(), c)
		return
	}

	// 调用服务更新字典
	err = api.dictService.UpdateDict(id, dictFormBO)
	if err != nil {
		common.FailWithMessage("修改字典失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("修改字典成功", c)
}

// DeleteDict 删除字典
// @Summary 删除字典
// @Description 删除字典
// @Tags 字典接口
// @Accept application/json
// @Produce application/json
// @Param ids path string true "字典ID，多个以英文逗号(,)拼接"
// @Success 200 {object} vo.Result
// @Router /api/v1/dict/{ids} [delete]
func (api *SysDictApi) DeleteDict(c *gin.Context) {
	// 获取路径参数
	ids := c.Param("ids")

	// 调用服务删除字典
	err := api.dictService.DeleteDict(ids)
	if err != nil {
		common.FailWithMessage("删除字典失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("删除字典成功", c)
}

// ListDictOptions 字典下拉列表
// @Summary 字典下拉列表
// @Description 获取字典下拉列表
// @Tags 字典接口
// @Accept application/json
// @Produce application/json
// @Param typeCode path string true "字典类型编码"
// @Success 200 {object} vo.Result{data=[]vo.Option}
// @Router /api/v1/dict/{typeCode}/options [get]
func (api *SysDictApi) ListDictOptions(c *gin.Context) {
	// 获取路径参数
	typeCode := c.Param("id")

	// 调用服务获取字典下拉列表
	options, err := api.dictService.ListDictOptions(typeCode)
	if err != nil {
		common.FailWithMessage("获取字典下拉列表失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(options, "查询成功", c)
}

// GetDictTypePage 字典类型分页列表
// @Summary 字典类型分页列表
// @Description 获取字典类型分页列表
// @Tags 字典接口
// @Accept application/json
// @Produce application/json
// @Param keywords query string false "关键字(类型名称/类型编码)"
// @Param pageNum query int false "页码"
// @Param pageSize query int false "每页条数"
// @Success 200 {object} vo.Result{data=vo.PageResult[vo.DictTypePageVO]}
// @Router /api/v1/dict/types/page [get]
func (api *SysDictApi) GetDictTypePage(c *gin.Context) {
	// 解析查询参数
	var queryParams query.DictTypePageQuery
	queryParams.Keywords = c.Query("keywords")

	if pageNumStr := c.Query("pageNum"); pageNumStr != "" {
		if pageNum, err := strconv.Atoi(pageNumStr); err == nil {
			queryParams.PageNum = pageNum
		} else {
			queryParams.PageNum = 1
		}
	} else {
		queryParams.PageNum = 1
	}

	if pageSizeStr := c.Query("pageSize"); pageSizeStr != "" {
		if pageSize, err := strconv.Atoi(pageSizeStr); err == nil {
			queryParams.PageSize = pageSize
		} else {
			queryParams.PageSize = 10
		}
	} else {
		queryParams.PageSize = 10
	}

	// 调用服务获取分页数据
	result, err := api.dictTypeService.GetDictTypePage(queryParams)
	if err != nil {
		common.FailWithMessage("获取字典类型分页列表失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(result, "查询成功", c)
}

// GetDictTypeForm 字典类型表单数据
// @Summary 字典类型表单数据
// @Description 获取字典类型表单数据
// @Tags 字典接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "字典ID"
// @Success 200 {object} vo.Result{data=bo.DictTypeFormBO}
// @Router /api/v1/dict/types/{id}/form [get]
func (api *SysDictApi) GetDictTypeForm(c *gin.Context) {
	// 获取路径参数
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		common.FailWithMessage("字典类型ID格式不正确", c)
		return
	}

	// 调用服务获取字典类型表单数据
	dictTypeFormBO, err := api.dictTypeService.GetDictTypeForm(id)
	if err != nil {
		common.FailWithMessage("获取字典类型表单数据失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(dictTypeFormBO, "查询成功", c)
}

// SaveDictType 新增字典类型
// @Summary 新增字典类型
// @Description 新增字典类型
// @Tags 字典接口
// @Accept application/json
// @Produce application/json
// @Param dictTypeForm body bo.DictTypeFormBO true "字典类型表单"
// @Success 200 {object} vo.Result
// @Router /api/v1/dict/types [post]
func (api *SysDictApi) SaveDictType(c *gin.Context) {
	// 绑定请求参数
	var dictTypeFormBO bo.DictTypeFormBO
	if err := c.ShouldBindJSON(&dictTypeFormBO); err != nil {
		common.FailWithMessage("请求参数解析失败: "+err.Error(), c)
		return
	}

	// 调用服务保存字典类型
	err := api.dictTypeService.SaveDictType(dictTypeFormBO)
	if err != nil {
		common.FailWithMessage("新增字典类型失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("新增字典类型成功", c)
}

// UpdateDictType 修改字典类型
// @Summary 修改字典类型
// @Description 修改字典类型
// @Tags 字典接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "字典类型ID"
// @Param dictTypeForm body bo.DictTypeFormBO true "字典类型表单"
// @Success 200 {object} vo.Result
// @Router /api/v1/dict/types/{id} [put]
func (api *SysDictApi) UpdateDictType(c *gin.Context) {
	// 获取路径参数
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		common.FailWithMessage("字典类型ID格式不正确", c)
		return
	}

	// 绑定请求参数
	var dictTypeFormBO bo.DictTypeFormBO
	if err := c.ShouldBindJSON(&dictTypeFormBO); err != nil {
		common.FailWithMessage("请求参数解析失败: "+err.Error(), c)
		return
	}

	// 调用服务更新字典类型
	err = api.dictTypeService.UpdateDictType(id, dictTypeFormBO)
	if err != nil {
		common.FailWithMessage("修改字典类型失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("修改字典类型成功", c)
}

// DeleteDictTypes 删除字典类型
// @Summary 删除字典类型
// @Description 删除字典类型
// @Tags 字典接口
// @Accept application/json
// @Produce application/json
// @Param ids path string true "字典类型ID，多个以英文逗号(,)分割"
// @Success 200 {object} vo.Result
// @Router /api/v1/dict/types/{ids} [delete]
func (api *SysDictApi) DeleteDictTypes(c *gin.Context) {
	// 获取路径参数
	ids := c.Param("ids")

	// 调用服务删除字典类型
	err := api.dictTypeService.DeleteDictTypes(ids)
	if err != nil {
		common.FailWithMessage("删除字典类型失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("删除字典类型成功", c)
}

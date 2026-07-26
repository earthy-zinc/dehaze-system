package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	dictservice "github.com/earthyzinc/dehaze-go/internal/service/dict"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/gin-gonic/gin"
)

type SysDictApi struct {
	dictService     dictservice.IDictService
	dictTypeService dictservice.IDictTypeService
}

func NewSysDictApi(dictService dictservice.IDictService, dictTypeService dictservice.IDictTypeService) *SysDictApi {
	return &SysDictApi{
		dictService:     dictService,
		dictTypeService: dictTypeService,
	}
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
// @Success 200 {object} common.Response{data=common.PageResult}
// @Router /api/v1/dict/page [get]
func (api *SysDictApi) GetDictPage(c *gin.Context) {
	// 解析查询参数
	var queryParams query.DictPageQuery
	queryParams.Keywords = c.Query("keywords")
	queryParams.TypeCode = c.Query("typeCode")
	queryParams.PageNum, queryParams.PageSize = getPageParams(c)

	// typeCode 必填校验
	if queryParams.TypeCode == "" {
		_ = c.Error(common.NewBizError(common.PARAM_IS_NULL, "字典类型编码不能为空"))
		return
	}

	// 调用服务获取分页数据
	result, err := api.dictService.GetPage(c.Request.Context(), &queryParams)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response{data=bo.DictFormBO}
// @Router /api/v1/dict/{id}/form [get]
func (api *SysDictApi) GetDictForm(c *gin.Context) {
	// 获取路径参数
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "字典ID格式不正确"))
		return
	}

	// 调用服务获取字典表单数据
	dictFormBO, err := api.dictService.GetFormData(c.Request.Context(), id)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response
// @Router /api/v1/dict [post]
func (api *SysDictApi) SaveDict(c *gin.Context) {
	// 绑定请求参数
	var dictFormBO bo.DictFormBO
	if err := c.ShouldBindJSON(&dictFormBO); err != nil {
		_ = c.Error(err)
		return
	}

	// 调用服务保存字典
	err := api.dictService.Create(c.Request.Context(), &dictFormBO)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response
// @Router /api/v1/dict/{id} [put]
func (api *SysDictApi) UpdateDict(c *gin.Context) {
	// 获取路径参数
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "字典ID格式不正确"))
		return
	}

	// 绑定请求参数
	var dictFormBO bo.DictFormBO
	if err := c.ShouldBindJSON(&dictFormBO); err != nil {
		_ = c.Error(err)
		return
	}

	// 调用服务更新字典
	err = api.dictService.Update(c.Request.Context(), id, &dictFormBO)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response
// @Router /api/v1/dict/{ids} [delete]
func (api *SysDictApi) DeleteDict(c *gin.Context) {
	ids, err := parseIDsFromCSV(c.Param("ids"))
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := api.dictService.Delete(c.Request.Context(), ids); err != nil {
		_ = c.Error(err)
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
// @Param id path string true "字典类型编码"
// @Success 200 {object} common.Response{data=[]vo.Option}
// @Router /api/v1/dict/{id}/options [get]
func (api *SysDictApi) ListDictOptions(c *gin.Context) {
	// 获取路径参数（路由参数名统一为 :id）
	typeCode := c.Param("id")

	// 调用服务获取字典下拉列表
	options, err := api.dictService.GetByTypeCode(c.Request.Context(), typeCode)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response{data=common.PageResult}
// @Router /api/v1/dict/types/page [get]
func (api *SysDictApi) GetDictTypePage(c *gin.Context) {
	// 解析查询参数
	var queryParams query.DictTypePageQuery
	queryParams.Keywords = c.Query("keywords")
	queryParams.PageNum, queryParams.PageSize = getPageParams(c)

	// 调用服务获取分页数据
	result, err := api.dictTypeService.GetPage(c.Request.Context(), &queryParams)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response{data=bo.DictTypeFormBO}
// @Router /api/v1/dict/types/{id}/form [get]
func (api *SysDictApi) GetDictTypeForm(c *gin.Context) {
	// 获取路径参数
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "字典类型ID格式不正确"))
		return
	}

	// 调用服务获取字典类型表单数据
	dictTypeFormBO, err := api.dictTypeService.GetFormData(c.Request.Context(), id)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response
// @Router /api/v1/dict/types [post]
func (api *SysDictApi) SaveDictType(c *gin.Context) {
	// 绑定请求参数
	var dictTypeFormBO bo.DictTypeFormBO
	if err := c.ShouldBindJSON(&dictTypeFormBO); err != nil {
		_ = c.Error(err)
		return
	}

	// 调用服务保存字典类型
	err := api.dictTypeService.Create(c.Request.Context(), &dictTypeFormBO)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response
// @Router /api/v1/dict/types/{id} [put]
func (api *SysDictApi) UpdateDictType(c *gin.Context) {
	// 获取路径参数
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "字典类型ID格式不正确"))
		return
	}

	// 绑定请求参数
	var dictTypeFormBO bo.DictTypeFormBO
	if err := c.ShouldBindJSON(&dictTypeFormBO); err != nil {
		_ = c.Error(err)
		return
	}

	// 调用服务更新字典类型
	err = api.dictTypeService.Update(c.Request.Context(), id, &dictTypeFormBO)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithMessage("修改字典类型成功", c)
}

// DeleteDictTypes 删除字典类型
// @Summary 删除字典类型
// @Description force=true 时级联删除关联的字典数据
// @Tags 字典接口
// @Accept application/json
// @Produce application/json
// @Param ids path string true "字典类型ID，多个以英文逗号(,)分割"
// @Param force query bool false "是否强制删除关联的字典数据"
// @Success 200 {object} common.Response
// @Router /api/v1/dict/types/{ids} [delete]
func (api *SysDictApi) DeleteDictTypes(c *gin.Context) {
	ids, err := parseIDsFromCSV(c.Param("ids"))
	if err != nil {
		_ = c.Error(err)
		return
	}
	force := c.Query("force") == "true"
	if err := api.dictTypeService.Delete(c.Request.Context(), ids, force); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("删除字典类型成功", c)
}

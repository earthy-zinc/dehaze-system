package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	deptservice "github.com/earthyzinc/dehaze-go/internal/service/dept"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/gin-gonic/gin"
)

type SysDeptApi struct {
	deptService deptservice.IDeptService
}

func NewSysDeptApi(deptService deptservice.IDeptService) *SysDeptApi {
	return &SysDeptApi{
		deptService: deptService,
	}
}

// ListDepartments 获取部门列表
// @Summary 获取部门列表
// @Description 获取部门列表
// @Tags 部门接口
// @Accept application/json
// @Produce application/json
// @Param keywords query string false "关键字(部门名称)"
// @Param status query int false "状态(1->正常；0->禁用)"
// @Success 200 {object} common.Response{data=[]vo.DeptVO}
// @Router /api/v1/depts [get]
func (api *SysDeptApi) ListDepartments(c *gin.Context) {
	// 解析查询参数
	var queryParams query.DeptQuery
	queryParams.Keywords = c.Query("keywords")
	if statusStr := c.Query("status"); statusStr != "" {
		if status, err := strconv.Atoi(statusStr); err == nil {
			queryParams.Status = &status
		}
	}

	// 调用服务获取部门列表
	ctx := c.Request.Context()
	deptVOs, err := api.deptService.GetList(ctx, &queryParams)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(deptVOs, "查询成功", c)
}

// ListDeptOptions 获取部门下拉选项
// @Summary 获取部门下拉选项
// @Description 获取部门下拉选项
// @Tags 部门接口
// @Accept application/json
// @Produce application/json
// @Success 200 {object} common.Response{data=[]vo.Option}
// @Router /api/v1/depts/options [get]
func (api *SysDeptApi) ListDeptOptions(c *gin.Context) {
	// 调用服务获取部门下拉选项
	ctx := c.Request.Context()
	options, err := api.deptService.GetOptions(ctx)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(options, "查询成功", c)
}

// GetDeptForm 获取部门表单数据
// @Summary 获取部门表单数据
// @Description 获取部门表单数据
// @Tags 部门接口
// @Accept application/json
// @Produce application/json
// @Param deptId path int true "部门ID"
// @Success 200 {object} common.Response{data=bo.DeptFormBO}
// @Router /api/v1/depts/{deptId}/form [get]
func (api *SysDeptApi) GetDeptForm(c *gin.Context) {
	// 获取路径参数
	deptIdStr := c.Param("deptId")
	deptId, err := strconv.ParseInt(deptIdStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "部门ID格式不正确"))
		return
	}

	// 调用服务获取部门表单数据
	ctx := c.Request.Context()
	deptFormBO, err := api.deptService.GetFormData(ctx, deptId)
	if err != nil {
		_ = c.Error(err)
		return
	}

	// 部门不存在时返回null（与Java行为一致）
	if deptFormBO == nil || deptFormBO.ID == nil {
		common.OkWithDetailed(nil, "查询成功", c)
		return
	}

	common.OkWithDetailed(deptFormBO, "查询成功", c)
}

// SaveDept 新增部门
// @Summary 新增部门
// @Description 新增部门
// @Tags 部门接口
// @Accept application/json
// @Produce application/json
// @Param formData body bo.DeptFormBO true "部门表单数据"
// @Success 200 {object} common.Response
// @Router /api/v1/depts [post]
func (api *SysDeptApi) SaveDept(c *gin.Context) {
	// 绑定请求参数
	var deptFormBO bo.DeptFormBO
	if err := c.ShouldBindJSON(&deptFormBO); err != nil {
		_ = c.Error(err)
		return
	}

	// 调用服务保存部门
	ctx := c.Request.Context()
	id, err := api.deptService.Create(ctx, &deptFormBO)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithData(id, c)
}

// UpdateDept 修改部门
// @Summary 修改部门
// @Description 修改部门
// @Tags 部门接口
// @Accept application/json
// @Produce application/json
// @Param deptId path int true "部门ID"
// @Param formData body bo.DeptFormBO true "部门表单数据"
// @Success 200 {object} common.Response
// @Router /api/v1/depts/{deptId} [put]
func (api *SysDeptApi) UpdateDept(c *gin.Context) {
	// 获取路径参数
	deptIdStr := c.Param("deptId")
	deptId, err := strconv.ParseInt(deptIdStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "部门ID格式不正确"))
		return
	}

	// 绑定请求参数
	var deptFormBO bo.DeptFormBO
	if err := c.ShouldBindJSON(&deptFormBO); err != nil {
		_ = c.Error(err)
		return
	}

	// 调用服务更新部门
	ctx := c.Request.Context()
	err = api.deptService.Update(ctx, deptId, &deptFormBO)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithMessage("修改部门成功", c)
}

// DeleteDepartments 删除部门
// @Summary 删除部门
// @Description 删除部门
// @Tags 部门接口
// @Accept application/json
// @Produce application/json
// @Param deptId path int true "部门ID"
// @Success 200 {object} common.Response
// @Router /api/v1/depts/{deptId} [delete]
func (api *SysDeptApi) DeleteDepartments(c *gin.Context) {
	// 获取路径参数
	deptIdStr := c.Param("deptId")
	deptId, err := strconv.ParseInt(deptIdStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "部门ID格式不正确"))
		return
	}

	// 调用服务删除部门
	ctx := c.Request.Context()
	err = api.deptService.Delete(ctx, deptId)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithMessage("删除部门成功", c)
}

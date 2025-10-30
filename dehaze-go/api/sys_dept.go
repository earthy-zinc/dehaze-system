package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/common"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/gin-gonic/gin"
)

type SysDeptApi struct {
	deptService service.DeptService
}

// ListDepartments 获取部门列表
// @Summary 获取部门列表
// @Description 获取部门列表
// @Tags 部门接口
// @Accept application/json
// @Produce application/json
// @Param keywords query string false "关键字(部门名称)"
// @Param status query int false "状态(1->正常；0->禁用)"
// @Success 200 {object} vo.Result{data=[]vo.DeptVO}
// @Router /api/v1/dept [get]
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
	deptVOs, err := api.deptService.ListDepartments(queryParams)
	if err != nil {
		common.FailWithMessage("获取部门列表失败: "+err.Error(), c)
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
// @Success 200 {object} vo.Result{data=[]vo.Option}
// @Router /api/v1/dept/options [get]
func (api *SysDeptApi) ListDeptOptions(c *gin.Context) {
	// 调用服务获取部门下拉选项
	options, err := api.deptService.ListDeptOptions()
	if err != nil {
		common.FailWithMessage("获取部门下拉选项失败: "+err.Error(), c)
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
// @Success 200 {object} vo.Result{data=bo.DeptFormBO}
// @Router /api/v1/dept/{deptId}/form [get]
func (api *SysDeptApi) GetDeptForm(c *gin.Context) {
	// 获取路径参数
	deptIdStr := c.Param("deptId")
	deptId, err := strconv.ParseInt(deptIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("部门ID格式不正确", c)
		return
	}

	// 调用服务获取部门表单数据
	deptFormBO, err := api.deptService.GetDeptForm(deptId)
	if err != nil {
		common.FailWithMessage("获取部门表单数据失败: "+err.Error(), c)
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
// @Success 200 {object} vo.Result{data=int64}
// @Router /api/v1/dept [post]
func (api *SysDeptApi) SaveDept(c *gin.Context) {
	// 绑定请求参数
	var deptFormBO bo.DeptFormBO
	if err := c.ShouldBindJSON(&deptFormBO); err != nil {
		common.FailWithMessage("请求参数解析失败: "+err.Error(), c)
		return
	}

	// 调用服务保存部门
	id, err := api.deptService.SaveDept(deptFormBO)
	if err != nil {
		common.FailWithMessage("新增部门失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(id, "新增部门成功", c)
}

// UpdateDept 修改部门
// @Summary 修改部门
// @Description 修改部门
// @Tags 部门接口
// @Accept application/json
// @Produce application/json
// @Param deptId path int true "部门ID"
// @Param formData body bo.DeptFormBO true "部门表单数据"
// @Success 200 {object} vo.Result{data=int64}
// @Router /api/v1/dept/{deptId} [put]
func (api *SysDeptApi) UpdateDept(c *gin.Context) {
	// 获取路径参数
	deptIdStr := c.Param("deptId")
	deptId, err := strconv.ParseInt(deptIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("部门ID格式不正确", c)
		return
	}

	// 绑定请求参数
	var deptFormBO bo.DeptFormBO
	if err := c.ShouldBindJSON(&deptFormBO); err != nil {
		common.FailWithMessage("请求参数解析失败: "+err.Error(), c)
		return
	}

	// 调用服务更新部门
	id, err := api.deptService.UpdateDept(deptId, deptFormBO)
	if err != nil {
		common.FailWithMessage("修改部门失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(id, "修改部门成功", c)
}

// DeleteDepartments 删除部门
// @Summary 删除部门
// @Description 删除部门
// @Tags 部门接口
// @Accept application/json
// @Produce application/json
// @Param ids path string true "部门ID，多个以英文逗号(,)分割"
// @Success 200 {object} vo.Result
// @Router /api/v1/dept/{ids} [delete]
func (api *SysDeptApi) DeleteDepartments(c *gin.Context) {
	// 获取路径参数
	ids := c.Param("ids")

	// 调用服务删除部门
	err := api.deptService.DeleteByIds(ids)
	if err != nil {
		common.FailWithMessage("删除部门失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("删除部门成功", c)
}

package api

import (
	"context"
	"strconv"
	"strings"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/service"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/gin-gonic/gin"
)

type SysRoleApi struct {
	roleService service.IRoleService
}

// GetRolePage 角色分页列表
// @Summary 角色分页列表
// @Description 获取角色分页列表
// @Tags 角色接口
// @Accept application/json
// @Produce application/json
// @Param keywords query string false "关键字(角色名称/角色编码)"
// @Param pageNum query int false "页码"
// @Param pageSize query int false "每页条数"
// @Success 200 {object} vo.Result{data=vo.PageResult[vo.RolePageVO]}
// @Router /api/v1/roles/page [get]
func (api *SysRoleApi) GetRolePage(c *gin.Context) {
	var _ context.Context
	ctx := c.Request.Context()

	// 解析查询参数
	var queryParams query.RolePageQuery
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
	result, err := api.roleService.GetPage(ctx, &queryParams)
	if err != nil {
		common.FailWithMessage("获取角色分页列表失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(result, "查询成功", c)
}

// ListRoleOptions 角色下拉列表
// @Summary 角色下拉列表
// @Description 获取角色下拉列表
// @Tags 角色接口
// @Accept application/json
// @Produce application/json
// @Success 200 {object} vo.Result{data=[]vo.Option}
// @Router /api/v1/roles/options [get]
func (api *SysRoleApi) ListRoleOptions(c *gin.Context) {
	ctx := c.Request.Context()

	// 调用服务获取角色下拉列表
	options, err := api.roleService.GetOptions(ctx)
	if err != nil {
		common.FailWithMessage("获取角色下拉列表失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(options, "查询成功", c)
}

// AddRole 新增角色
// @Summary 新增角色
// @Description 新增角色
// @Tags 角色接口
// @Accept application/json
// @Produce application/json
// @Param roleForm body bo.RoleFormBO true "角色表单对象"
// @Success 200 {object} vo.Result
// @Router /api/v1/roles [post]
func (api *SysRoleApi) AddRole(c *gin.Context) {
	ctx := c.Request.Context()

	// 绑定请求参数
	var roleFormBO bo.RoleFormBO
	if err := c.ShouldBindJSON(&roleFormBO); err != nil {
		common.FailWithMessage("请求参数解析失败: "+err.Error(), c)
		return
	}

	// 调用服务保存角色
	err := api.roleService.Create(ctx, &roleFormBO)
	if err != nil {
		common.FailWithMessage("新增角色失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("新增角色成功", c)
}

// GetRoleForm 角色表单数据
// @Summary 角色表单数据
// @Description 获取角色表单数据
// @Tags 角色接口
// @Accept application/json
// @Produce application/json
// @Param roleId path int true "角色ID"
// @Success 200 {object} vo.Result{data=bo.RoleFormBO}
// @Router /api/v1/roles/{roleId}/form [get]
func (api *SysRoleApi) GetRoleForm(c *gin.Context) {
	ctx := c.Request.Context()

	// 获取路径参数
	roleIdStr := c.Param("roleId")
	roleId, err := strconv.ParseInt(roleIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("角色ID格式不正确", c)
		return
	}

	// 调用服务获取角色表单数据
	roleFormBO, err := api.roleService.GetFormData(ctx, roleId)
	if err != nil {
		common.FailWithMessage("获取角色表单数据失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(roleFormBO, "查询成功", c)
}

// UpdateRole 修改角色
// @Summary 修改角色
// @Description 修改角色
// @Tags 角色接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "角色ID"
// @Param roleForm body bo.RoleFormBO true "角色表单对象"
// @Success 200 {object} vo.Result
// @Router /api/v1/roles/{id} [put]
func (api *SysRoleApi) UpdateRole(c *gin.Context) {
	ctx := c.Request.Context()

	// 获取路径参数
	idStr := c.Param("roleId")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		common.FailWithMessage("角色ID格式不正确", c)
		return
	}

	// 绑定请求参数
	var roleFormBO bo.RoleFormBO
	if err := c.ShouldBindJSON(&roleFormBO); err != nil {
		common.FailWithMessage("请求参数解析失败: "+err.Error(), c)
		return
	}

	// 调用服务保存角色
	err = api.roleService.Update(ctx, id, &roleFormBO)
	if err != nil {
		common.FailWithMessage("修改角色失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("修改角色成功", c)
}

// DeleteRoles 删除角色
// @Summary 删除角色
// @Description 删除角色
// @Tags 角色接口
// @Accept application/json
// @Produce application/json
// @Param ids path string true "删除角色，多个以英文逗号(,)拼接"
// @Success 200 {object} vo.Result
// @Router /api/v1/roles/{ids} [delete]
func (api *SysRoleApi) DeleteRoles(c *gin.Context) {
	ctx := c.Request.Context()

	// 获取路径参数
	ids := c.Param("ids")

	// 解析ID列表
	idStrings := strings.Split(ids, ",")
	var idList []int64
	for _, idStr := range idStrings {
		id, err := strconv.ParseInt(strings.TrimSpace(idStr), 10, 64)
		if err != nil {
			common.FailWithMessage("角色ID格式不正确", c)
			return
		}
		idList = append(idList, id)
	}

	// 调用服务删除角色
	err := api.roleService.Delete(ctx, idList)
	if err != nil {
		common.FailWithMessage("删除角色失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("删除角色成功", c)
}

// UpdateRoleStatus 修改角色状态
// @Summary 修改角色状态
// @Description 修改角色状态
// @Tags 角色接口
// @Accept application/json
// @Produce application/json
// @Param roleId path int true "角色ID"
// @Param status query int true "状态(1:启用;0:禁用)"
// @Success 200 {object} vo.Result
// @Router /api/v1/roles/{roleId}/status [put]
func (api *SysRoleApi) UpdateRoleStatus(c *gin.Context) {
	ctx := c.Request.Context()

	// 获取路径参数
	roleIdStr := c.Param("roleId")
	roleId, err := strconv.ParseInt(roleIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("角色ID格式不正确", c)
		return
	}

	// 获取查询参数
	statusStr := c.Query("status")
	status, err := strconv.Atoi(statusStr)
	if err != nil {
		common.FailWithMessage("状态参数格式不正确", c)
		return
	}

	// 调用服务更新角色状态
	err = api.roleService.UpdateStatus(ctx, roleId, int8(status))
	if err != nil {
		common.FailWithMessage("修改角色状态失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("修改角色状态成功", c)
}

// GetRoleMenuIds 获取角色的菜单ID集合
// @Summary 获取角色的菜单ID集合
// @Description 获取角色的菜单ID集合
// @Tags 角色接口
// @Accept application/json
// @Produce application/json
// @Param roleId path int true "角色ID"
// @Success 200 {object} vo.Result{data=[]int64}
// @Router /api/v1/roles/{roleId}/menuIds [get]
func (api *SysRoleApi) GetRoleMenuIds(c *gin.Context) {
	ctx := c.Request.Context()

	// 获取路径参数
	roleIdStr := c.Param("roleId")
	roleId, err := strconv.ParseInt(roleIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("角色ID格式不正确", c)
		return
	}

	// 调用服务获取角色菜单ID集合
	menuIds, err := api.roleService.GetMenuIDs(ctx, roleId)
	if err != nil {
		common.FailWithMessage("获取角色菜单ID集合失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(menuIds, "查询成功", c)
}

// AssignMenusToRole 分配菜单(包括按钮权限)给角色
// @Summary 分配菜单(包括按钮权限)给角色
// @Description 分配菜单(包括按钮权限)给角色
// @Tags 角色接口
// @Accept application/json
// @Produce application/json
// @Param roleId path int true "角色ID"
// @Param menuIds body []int64 true "菜单ID列表"
// @Success 200 {object} vo.Result
// @Router /api/v1/roles/{roleId}/menus [put]
func (api *SysRoleApi) AssignMenusToRole(c *gin.Context) {
	ctx := c.Request.Context()

	// 获取路径参数
	roleIdStr := c.Param("roleId")
	roleId, err := strconv.ParseInt(roleIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("角色ID格式不正确", c)
		return
	}

	// 绑定请求参数
	var menuIds []int64
	if err := c.ShouldBindJSON(&menuIds); err != nil {
		common.FailWithMessage("请求参数解析失败: "+err.Error(), c)
		return
	}

	// 调用服务分配菜单给角色
	err = api.roleService.AssignMenus(ctx, roleId, menuIds)
	if err != nil {
		common.FailWithMessage("分配菜单给角色失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("分配菜单给角色成功", c)
}

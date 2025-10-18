package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/common"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/gin-gonic/gin"
)

type SysMenuApi struct {
	menuService service.MenuService
}

// ListMenus 菜单列表
// @Summary 菜单列表
// @Description 获取菜单列表
// @Tags 菜单接口
// @Accept application/json
// @Produce application/json
// @Param keywords query string false "关键字(菜单名称)"
// @Param status query int false "状态(1->显示；0->隐藏)"
// @Success 200 {object} vo.Result{data=[]vo.MenuVO}
// @Router /api/v1/menus [get]
func (api *SysMenuApi) ListMenus(c *gin.Context) {
	// 解析查询参数
	var queryParams query.MenuQuery
	queryParams.Keywords = c.Query("keywords")
	
	if statusStr := c.Query("status"); statusStr != "" {
		if status, err := strconv.Atoi(statusStr); err == nil {
			queryParams.Status = &status
		}
	}

	// 调用服务获取菜单列表
	menuList, err := api.menuService.ListMenus(queryParams)
	if err != nil {
		common.FailWithMessage("获取菜单列表失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(menuList, "查询成功", c)
}

// ListMenuOptions 菜单下拉列表
// @Summary 菜单下拉列表
// @Description 获取菜单下拉列表
// @Tags 菜单接口
// @Accept application/json
// @Produce application/json
// @Success 200 {object} vo.Result{data=[]vo.Option}
// @Router /api/v1/menus/options [get]
func (api *SysMenuApi) ListMenuOptions(c *gin.Context) {
	// 调用服务获取菜单下拉列表
	options, err := api.menuService.ListMenuOptions()
	if err != nil {
		common.FailWithMessage("获取菜单下拉列表失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(options, "查询成功", c)
}

// ListRoutes 路由列表
// @Summary 路由列表
// @Description 获取路由列表
// @Tags 菜单接口
// @Accept application/json
// @Produce application/json
// @Success 200 {object} vo.Result{data=[]vo.RouteVO}
// @Router /api/v1/menus/routes [get]
func (api *SysMenuApi) ListRoutes(c *gin.Context) {
	// 调用服务获取路由列表
	routes, err := api.menuService.ListRoutes()
	if err != nil {
		common.FailWithMessage("获取路由列表失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(routes, "查询成功", c)
}

// GetMenuForm 菜单表单数据
// @Summary 菜单表单数据
// @Description 获取菜单表单数据
// @Tags 菜单接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "菜单ID"
// @Success 200 {object} vo.Result{data=bo.MenuForm}
// @Router /api/v1/menus/{id}/form [get]
func (api *SysMenuApi) GetMenuForm(c *gin.Context) {
	// 获取路径参数
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		common.FailWithMessage("菜单ID格式不正确", c)
		return
	}

	// 调用服务获取菜单表单数据
	menuForm, err := api.menuService.GetMenuForm(id)
	if err != nil {
		common.FailWithMessage("获取菜单表单数据失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(menuForm, "查询成功", c)
}

// SaveMenu 新增菜单
// @Summary 新增菜单
// @Description 新增菜单
// @Tags 菜单接口
// @Accept application/json
// @Produce application/json
// @Param menuForm body bo.MenuForm true "菜单表单数据"
// @Success 200 {object} vo.Result
// @Router /api/v1/menus [post]
func (api *SysMenuApi) SaveMenu(c *gin.Context) {
	// 绑定请求参数
	var menuForm bo.MenuForm
	if err := c.ShouldBindJSON(&menuForm); err != nil {
		common.FailWithMessage("请求参数解析失败: "+err.Error(), c)
		return
	}

	// 调用服务保存菜单
	err := api.menuService.SaveMenu(menuForm)
	if err != nil {
		common.FailWithMessage("新增菜单失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("新增菜单成功", c)
}

// UpdateMenu 修改菜单
// @Summary 修改菜单
// @Description 修改菜单
// @Tags 菜单接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "菜单ID"
// @Param menuForm body bo.MenuForm true "菜单表单数据"
// @Success 200 {object} vo.Result
// @Router /api/v1/menus/{id} [put]
func (api *SysMenuApi) UpdateMenu(c *gin.Context) {
	// 获取路径参数
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		common.FailWithMessage("菜单ID格式不正确", c)
		return
	}

	// 绑定请求参数
	var menuForm bo.MenuForm
	if err := c.ShouldBindJSON(&menuForm); err != nil {
		common.FailWithMessage("请求参数解析失败: "+err.Error(), c)
		return
	}

	// 设置菜单ID
	menuForm.ID = &id

	// 调用服务更新菜单
	err = api.menuService.SaveMenu(menuForm)
	if err != nil {
		common.FailWithMessage("修改菜单失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("修改菜单成功", c)
}

// DeleteMenu 删除菜单
// @Summary 删除菜单
// @Description 删除菜单
// @Tags 菜单接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "菜单ID"
// @Success 200 {object} vo.Result
// @Router /api/v1/menus/{id} [delete]
func (api *SysMenuApi) DeleteMenu(c *gin.Context) {
	// 获取路径参数
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		common.FailWithMessage("菜单ID格式不正确", c)
		return
	}

	// 调用服务删除菜单
	err = api.menuService.DeleteMenu(id)
	if err != nil {
		common.FailWithMessage("删除菜单失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("删除菜单成功", c)
}

// UpdateMenuVisible 修改菜单显示状态
// @Summary 修改菜单显示状态
// @Description 修改菜单显示状态
// @Tags 菜单接口
// @Accept application/json
// @Produce application/json
// @Param menuId path int true "菜单ID"
// @Param visible query int true "显示状态(1:显示;0:隐藏)"
// @Success 200 {object} vo.Result
// @Router /api/v1/menus/{menuId} [patch]
func (api *SysMenuApi) UpdateMenuVisible(c *gin.Context) {
	// 获取路径参数
	menuIdStr := c.Param("menuId")
	menuId, err := strconv.ParseInt(menuIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("菜单ID格式不正确", c)
		return
	}

	// 获取查询参数
	visibleStr := c.Query("visible")
	visible, err := strconv.Atoi(visibleStr)
	if err != nil {
		common.FailWithMessage("显示状态参数格式不正确", c)
		return
	}

	// 调用服务更新菜单显示状态
	err = api.menuService.UpdateMenuVisible(menuId, visible)
	if err != nil {
		common.FailWithMessage("修改菜单显示状态失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("修改菜单显示状态成功", c)
}
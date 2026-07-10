package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/service/menu"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

type SysMenuApi struct {
	menuService *menu.MenuService
}

func NewSysMenuApi(menuService *menu.MenuService) *SysMenuApi {
	return &SysMenuApi{
		menuService: menuService,
	}
}

// ListMenus 菜单列表
// @Summary 菜单列表
// @Description 获取菜单列表
// @Tags 菜单接口
// @Accept application/json
// @Produce application/json
// @Param keywords query string false "关键字(菜单名称)"
// @Param status query int false "状态(1->显示；0->隐藏)"
// @Success 200 {object} common.Response{data=[]vo.MenuVO}
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
	ctx := c.Request.Context()
	menuList, err := api.menuService.GetList(ctx, &queryParams)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response{data=[]vo.Option}
// @Router /api/v1/menus/options [get]
func (api *SysMenuApi) ListMenuOptions(c *gin.Context) {
	// 调用服务获取菜单下拉列表
	ctx := c.Request.Context()
	options, err := api.menuService.GetOptions(ctx)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response{data=[]vo.RouteVO}
// @Router /api/v1/menus/routes [get]
func (api *SysMenuApi) ListRoutes(c *gin.Context) {
	// 从上下文获取当前用户角色
	claims := security.GetUserInfo(c)
	roles := []string{}
	if claims != nil {
		roles = claims.Authorities
	}

	// 调用服务获取路由列表
	ctx := c.Request.Context()
	routes, err := api.menuService.GetRoutes(ctx, roles)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response{data=bo.MenuForm}
// @Router /api/v1/menus/{id}/form [get]
func (api *SysMenuApi) GetMenuForm(c *gin.Context) {
	// 获取路径参数
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "菜单ID格式不正确"))
		return
	}

	// 调用服务获取菜单表单数据
	ctx := c.Request.Context()
	menuForm, err := api.menuService.GetFormData(ctx, id)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response
// @Router /api/v1/menus [post]
func (api *SysMenuApi) SaveMenu(c *gin.Context) {
	// 绑定请求参数
	var menuForm bo.MenuForm
	if err := c.ShouldBindJSON(&menuForm); err != nil {
		_ = c.Error(err)
		return
	}

	// 调用服务保存菜单
	ctx := c.Request.Context()
	err := api.menuService.Create(ctx, &menuForm)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response
// @Router /api/v1/menus/{id} [put]
func (api *SysMenuApi) UpdateMenu(c *gin.Context) {
	// 获取路径参数
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "菜单ID格式不正确"))
		return
	}

	// 绑定请求参数
	var menuForm bo.MenuForm
	if err := c.ShouldBindJSON(&menuForm); err != nil {
		_ = c.Error(err)
		return
	}

	// 调用服务更新菜单
	ctx := c.Request.Context()
	err = api.menuService.Update(ctx, id, &menuForm)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response
// @Router /api/v1/menus/{id} [delete]
func (api *SysMenuApi) DeleteMenu(c *gin.Context) {
	// 获取路径参数
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "菜单ID格式不正确"))
		return
	}

	// 调用服务删除菜单
	ctx := c.Request.Context()
	err = api.menuService.Delete(ctx, id)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response
// @Router /api/v1/menus/{menuId} [patch]
func (api *SysMenuApi) UpdateMenuVisible(c *gin.Context) {
	// 获取路径参数
	menuIdStr := c.Param("menuId")
	menuId, err := strconv.ParseInt(menuIdStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "菜单ID格式不正确"))
		return
	}

	// 获取查询参数
	visibleStr := c.Query("visible")
	visible, err := strconv.Atoi(visibleStr)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "显示状态参数格式不正确"))
		return
	}

	// 校验visible值范围
	if visible != 0 && visible != 1 {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "显示状态参数值必须为0或1"))
		return
	}

	// 调用服务更新菜单显示状态
	ctx := c.Request.Context()
	err = api.menuService.UpdateMenuVisible(ctx, menuId, visible)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithMessage("修改菜单显示状态成功", c)
}

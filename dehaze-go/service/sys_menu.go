package service

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/model/vo"
	"github.com/earthyzinc/dehaze-go/utils"
	"gorm.io/gorm"
)

type MenuService struct{}

// ListMenus 菜单列表
func (menuService *MenuService) ListMenus(queryParams query.MenuQuery) (menuList []vo.MenuVO, err error) {
	var menus []model.SysMenu
	db := global.DB.Model(&model.SysMenu{})

	// 添加查询条件
	if queryParams.Keywords != "" {
		keyword := "%" + queryParams.Keywords + "%"
		db = db.Where("name LIKE ?", keyword)
	}
	if queryParams.Status != nil {
		db = db.Where("visible = ?", *queryParams.Status)
	}

	// 按排序字段升序排列
	err = db.Order("sort ASC").Find(&menus).Error
	if err != nil {
		return nil, err
	}

	// 构建菜单树
	menuList = buildMenuTree(0, menus)
	return menuList, nil
}

// ListMenuOptions 菜单下拉数据
func (menuService *MenuService) ListMenuOptions() (options []vo.Option, err error) {
	var menuList []model.SysMenu
	err = global.DB.Model(&model.SysMenu{}).
		Order("sort ASC").
		Find(&menuList).Error

	if err != nil {
		return nil, err
	}

	// 构建菜单下拉选项树
	options = buildMenuOptions(0, menuList)
	return options, nil
}

// ListRoutes 获取路由列表
func (menuService *MenuService) ListRoutes() (routes []vo.RouteVO, err error) {
	var routeBOs []bo.RouteBO
	err = global.DB.Model(&model.SysMenu{}).
		Select("sys_menu.id, sys_menu.parent_id, sys_menu.name, sys_menu.path, sys_menu.component, sys_menu.icon, sys_menu.sort, sys_menu.visible, sys_menu.redirect, sys_menu.type, sys_menu.always_show, sys_menu.keep_alive, sys_role.code").
		Joins("LEFT JOIN sys_role_menu ON sys_menu.id = sys_role_menu.menu_id").
		Joins("LEFT JOIN sys_role ON sys_role_menu.role_id = sys_role.id").
		Where("sys_menu.type != ?", 4). // 不包括按钮类型
		Order("sys_menu.sort ASC").
		Find(&routeBOs).Error

	if err != nil {
		return nil, err
	}

	// 构建路由树
	routes = buildRoutes(0, routeBOs)
	return routes, nil
}

// SaveMenu 新增/修改菜单
func (menuService *MenuService) SaveMenu(menuForm bo.MenuForm) (err error) {
	// 根据菜单类型处理特殊逻辑
	menuType := menuForm.Type

	if menuType == 2 { // 如果是目录
		path := menuForm.Path
		if menuForm.ParentID == 0 && !strings.HasPrefix(path, "/") {
			menuForm.Path = "/" + path // 一级目录需以 / 开头
		}
		menuForm.Component = "Layout"
	} else if menuType == 3 { // 如果是外链
		menuForm.Component = ""
	}

	// 生成树路径
	treePath := menuService.generateMenuTreePath(menuForm.ParentID)

	// 构建菜单实体
	menu := model.SysMenu{
		ParentID:   menuForm.ParentID,
		Name:       menuForm.Name,
		Type:       menuForm.Type,
		Path:       menuForm.Path,
		Component:  menuForm.Component,
		Perm:       menuForm.Perm,
		Visible:    int8(menuForm.Visible),
		Sort:       menuForm.Sort,
		Icon:       menuForm.Icon,
		Redirect:   menuForm.Redirect,
		TreePath:   treePath,
		AlwaysShow: int8(menuForm.AlwaysShow),
		KeepAlive:  int8(menuForm.KeepAlive),
		BaseModel: model.BaseModel{
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		},
	}

	// 如果ID存在则更新，否则创建
	if menuForm.ID != nil {
		menu.ID = *menuForm.ID
		menu.BaseModel.UpdatedAt = time.Now()
		err = global.DB.Save(&menu).Error
	} else {
		err = global.DB.Create(&menu).Error
	}

	// 更新菜单后清空所有角色权限缓存
	if err == nil {
		menuService.clearAllRolePermsCache()
	}

	return err
}

// DeleteMenu 删除菜单
func (menuService *MenuService) DeleteMenu(id int64) (err error) {
	// 删除菜单及其子菜单 - 修复SQL注入风险
	err = global.DB.Where("id = ? OR CONCAT(',',tree_path,',') LIKE CONCAT('%,',?,',%')", id, id).
		Delete(&model.SysMenu{}).
		Error

	// 删除成功后清空所有角色权限缓存
	if err == nil {
		menuService.clearAllRolePermsCache()
	}

	return err
}

// clearAllRolePermsCache 清空所有角色权限缓存
func (menuService *MenuService) clearAllRolePermsCache() {
	if global.REDIS == nil {
		return
	}

	ctx := context.Background()
	// 删除整个role:perms哈希表
	global.REDIS.Del(ctx, "role:perms")
}

// UpdateMenuVisible 修改菜单显示状态
func (menuService *MenuService) UpdateMenuVisible(menuId int64, visible int) (err error) {
	err = global.DB.Model(&model.SysMenu{}).
		Where("id = ?", menuId).
		Update("visible", visible).
		Error
	return err
}

// GetMenuForm 获取菜单表单数据
func (menuService *MenuService) GetMenuForm(id int64) (menuForm bo.MenuForm, err error) {
	var entity model.SysMenu
	err = global.DB.Where("id = ?", id).First(&entity).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return menuForm, errors.New("菜单不存在")
		}
		return menuForm, err
	}

	idPtr := entity.ID
	menuForm = bo.MenuForm{
		ID:         &idPtr,
		ParentID:   entity.ParentID,
		Name:       entity.Name,
		Type:       entity.Type,
		Path:       entity.Path,
		Component:  entity.Component,
		Perm:       entity.Perm,
		Visible:    int(entity.Visible),
		Sort:       entity.Sort,
		Icon:       entity.Icon,
		Redirect:   entity.Redirect,
		AlwaysShow: int(entity.AlwaysShow),
		KeepAlive:  int(entity.KeepAlive),
	}

	return menuForm, nil
}

// ListRolePerms 获取角色权限集合
func (menuService *MenuService) ListRolePerms(roles []string) (perms []string, err error) {
	if len(roles) == 0 {
		return []string{}, nil
	}

	err = global.DB.Model(&model.SysMenu{}).
		Select("DISTINCT sys_menu.perm").
		Joins("INNER JOIN sys_role_menu ON sys_menu.id = sys_role_menu.menu_id").
		Joins("INNER JOIN sys_role ON sys_role.id = sys_role_menu.role_id").
		Where("sys_menu.type = ?", 4). // 按钮类型
		Where("sys_menu.perm IS NOT NULL AND sys_menu.perm != ?", "").
		Where("sys_role.code IN ?", roles).
		Pluck("sys_menu.perm", &perms).
		Error

	return perms, err
}

// generateMenuTreePath 部门路径生成
func (menuService *MenuService) generateMenuTreePath(parentId int64) string {
	if parentId == 0 {
		return "0"
	} else {
		var parent model.SysMenu
		err := global.DB.Where("id = ?", parentId).First(&parent).Error
		if err != nil {
			return "0"
		}
		return parent.TreePath + "," + strconv.FormatInt(parent.ID, 10)
	}
}

// buildMenuTree 递归生成菜单列表
func buildMenuTree(parentId int64, menuList []model.SysMenu) []vo.MenuVO {
	var menuVOs []vo.MenuVO
	for _, menu := range menuList {
		if menu.ParentID == parentId {
			children := buildMenuTree(menu.ID, menuList)
			menuVO := vo.MenuVO{
				ID:        menu.ID,
				ParentID:  menu.ParentID,
				Name:      menu.Name,
				Path:      menu.Path,
				Component: menu.Component,
				Sort:      menu.Sort,
				Visible:   int(menu.Visible),
				Icon:      menu.Icon,
				Redirect:  menu.Redirect,
				Perm:      menu.Perm,
				Children:  children,
			}
			menuVOs = append(menuVOs, menuVO)
		}
	}
	return menuVOs
}

// buildMenuOptions 递归生成菜单下拉层级列表
func buildMenuOptions(parentId int64, menuList []model.SysMenu) []vo.Option {
	var options []vo.Option
	for _, menu := range menuList {
		if menu.ParentID == parentId {
			option := vo.Option{
				Value: menu.ID,
				Label: menu.Name,
			}
			children := buildMenuOptions(menu.ID, menuList)
			if len(children) > 0 {
				option.Children = children
			}
			options = append(options, option)
		}
	}
	return options
}

// buildRoutes 递归生成菜单路由层级列表
func buildRoutes(parentId int64, routeList []bo.RouteBO) []vo.RouteVO {
	var routes []vo.RouteVO
	for _, route := range routeList {
		if route.ParentID == parentId {
			// 构建Meta信息
			meta := vo.RouteMeta{
				Title:  route.Name,
				Icon:   route.Icon,
				Roles:  route.Roles,
				Hidden: route.Visible == 0,
			}

			// 【菜单】是否开启页面缓存
			if route.Type == 1 && route.KeepAlive == 1 {
				keepAlive := true
				meta.KeepAlive = &keepAlive
			}

			// 【目录】只有一个子路由是否始终显示
			if route.Type == 2 && route.AlwaysShow == 1 {
				alwaysShow := true
				meta.AlwaysShow = &alwaysShow
			}

			routeVO := vo.RouteVO{
				Name:      utils.ToCamelCase(route.Path), // 路由 name 需要驼峰，首字母大写
				Path:      route.Path,
				Redirect:  route.Redirect,
				Component: route.Component,
				Meta:      meta,
			}

			children := buildRoutes(route.ID, routeList)
			if len(children) > 0 {
				routeVO.Children = children
			}

			routes = append(routes, routeVO)
		}
	}
	return routes
}

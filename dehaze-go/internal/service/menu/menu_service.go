package menu

import (
	"context"
	"encoding/json"
	"errors"
	"strconv"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/enum"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	menurepo "github.com/earthyzinc/dehaze-go/internal/repository/menu"
	rolerepo "github.com/earthyzinc/dehaze-go/internal/repository/role"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"gorm.io/gorm"
)

// ROUTE_CACHE_KEY Redis中路由缓存key
const ROUTE_CACHE_KEY = "menu::routes"

// ROUTE_CACHE_EXPIRATION 路由缓存过期时间（1小时）
const ROUTE_CACHE_EXPIRATION = time.Hour

type MenuService struct {
	cache types.ICache

	menuRepo menurepo.IMenuRepository
	roleRepo rolerepo.IRoleRepository
}

func NewMenuService(cache types.ICache, menuRepo menurepo.IMenuRepository, roleRepo rolerepo.IRoleRepository) *MenuService {
	return &MenuService{cache: cache, menuRepo: menuRepo, roleRepo: roleRepo}
}

func (s *MenuService) GetList(ctx context.Context, q *query.MenuQuery) ([]vo.MenuVO, error) {
	if q == nil {
		q = &query.MenuQuery{}
	}

	menus, err := s.menuRepo.FindAll(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询菜单列表失败", err)
	}

	if len(menus) == 0 {
		return []vo.MenuVO{}, nil
	}

	// 对齐 Java TreeDataUtils.findRootIds：
	// 收集结果集中的所有ID，父ID不在ID集合中的即为根
	idSet := make(map[int64]bool, len(menus))
	for _, menu := range menus {
		idSet[menu.ID] = true
	}

	var rootIds []int64
	for _, menu := range menus {
		if !idSet[menu.ParentID] {
			rootIds = append(rootIds, menu.ParentID)
		}
	}

	// 去重 rootIds
	rootIdSet := make(map[int64]bool)
	for _, rootId := range rootIds {
		rootIdSet[rootId] = true
	}

	// 按 ParentID 分组，O(n) 构建树形结构
	childrenMap := make(map[int64][]model.SysMenu)
	for _, menu := range menus {
		childrenMap[menu.ParentID] = append(childrenMap[menu.ParentID], menu)
	}

	result := make([]vo.MenuVO, 0)
	for rootId := range rootIdSet {
		for _, menu := range childrenMap[rootId] {
			result = append(result, buildMenuVO(menu, childrenMap))
		}
	}
	return result, nil
}

func (s *MenuService) GetFormData(ctx context.Context, id int64) (*bo.MenuForm, error) {
	form, err := s.menuRepo.GetFormData(ctx, id)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "菜单不存在")
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询菜单失败", err)
	}
	if form == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "菜单不存在")
	}
	return form, nil
}

func (s *MenuService) Create(ctx context.Context, form *bo.MenuForm) error {
	if form == nil {
		return common.NewBizError(common.PARAM_ERROR, "表单数据不能为空")
	}

	menuType := form.Type
	path := form.Path
	// 目录类型：设置默认Layout组件
	if menuType == enum.MenuTypeCatalog {
		if form.ParentID == 0 && !strings.HasPrefix(path, "/") {
			path = "/" + path
		}
		form.Component = "Layout"
	} else if menuType == enum.MenuTypeExtlink {
		// 外链类型：清空组件路径
		form.Component = ""
	}

	// 业务校验
	if err := s.validateMenuForm(ctx, form, 0); err != nil {
		return err
	}

	treePath := s.generateMenuTreePath(ctx, form.ParentID)

	menu := &model.SysMenu{
		ParentID:   form.ParentID,
		Name:       form.Name,
		Type:       int8(form.Type),
		Path:       path,
		Component:  form.Component,
		Perm:       form.Perm,
		Visible:    int8(form.Visible),
		Sort:       form.Sort,
		Icon:       form.Icon,
		Redirect:   form.Redirect,
		TreePath:   treePath,
		AlwaysShow: int8(form.AlwaysShow),
		KeepAlive:  int8(form.KeepAlive),
		BaseModel: model.BaseModel{
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		},
	}

	if err := s.menuRepo.Create(ctx, menu); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "创建菜单失败", err)
	}

	// 新增菜单默认分配给超级管理员角色
	if rootRole, err := s.roleRepo.FindByCode(ctx, "ROOT"); err == nil && rootRole != nil {
		_ = s.menuRepo.SaveRoleMenu(ctx, rootRole.ID, menu.ID)
	}

	s.clearAllRolePermsCache(ctx)
	return nil
}

func (s *MenuService) Update(ctx context.Context, id int64, form *bo.MenuForm) error {
	if form == nil {
		return common.NewBizError(common.PARAM_ERROR, "表单数据不能为空")
	}

	// 检查菜单是否存在
	existingMenu, err := s.menuRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询菜单失败", err)
	}
	if existingMenu == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "菜单不存在")
	}

	menuType := form.Type
	path := form.Path

	// 目录类型：设置默认Layout组件
	if menuType == enum.MenuTypeCatalog {
		if form.ParentID == 0 && !strings.HasPrefix(path, "/") {
			path = "/" + path
		}
		form.Component = "Layout"
	} else if menuType == enum.MenuTypeExtlink {
		// 外链类型：清空组件路径
		form.Component = ""
	}

	// 业务校验（排除自身ID）
	if err := s.validateMenuForm(ctx, form, id); err != nil {
		return err
	}

	treePath := s.generateMenuTreePath(ctx, form.ParentID)

	menu := &model.SysMenu{
		BaseModel:  model.BaseModel{ID: id},
		ParentID:   form.ParentID,
		Name:       form.Name,
		Type:       int8(form.Type),
		Path:       path,
		Component:  form.Component,
		Perm:       form.Perm,
		Visible:    int8(form.Visible),
		Sort:       form.Sort,
		Icon:       form.Icon,
		Redirect:   form.Redirect,
		TreePath:   treePath,
		AlwaysShow: int8(form.AlwaysShow),
		KeepAlive:  int8(form.KeepAlive),
	}

	if err := s.menuRepo.Update(ctx, menu); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新菜单失败", err)
	}

	s.clearAllRolePermsCache(ctx)
	return nil
}

func (s *MenuService) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return nil
	}

	// 校验所有传入的菜单ID都存在
	count, err := s.menuRepo.CountByIDs(ctx, ids)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询菜单失败", err)
	}
	if count != int64(len(ids)) {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "菜单不存在")
	}

	// 使用事务包装删除操作，确保数据一致性
	err = s.menuRepo.Transaction(ctx, func(txRepo menurepo.IMenuRepository) error {
		// 先删除角色-菜单关联关系（含所有传入菜单及子孙菜单的关联，须在菜单删除前执行，否则子查询找不到菜单）
		if delErr := txRepo.DeleteRoleMenuByMenuIDs(ctx, ids); delErr != nil {
			return delErr
		}

		// 批量级联删除：删除所有传入菜单及其子孙菜单
		if _, delErr := txRepo.DeleteCascadeByIDs(ctx, ids); delErr != nil {
			return delErr
		}
		return nil
	})
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除菜单失败", err)
	}

	// 刷新权限缓存（同时会清除路由缓存）
	s.clearAllRolePermsCache(ctx)

	return nil
}

func (s *MenuService) GetOptions(ctx context.Context) ([]vo.Option, error) {
	options, err := s.menuRepo.GetMenuOptions(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询菜单列表失败", err)
	}

	// 按 ParentID 分组，O(n) 构建树形结构
	childrenMap := make(map[int64][]read.MenuOptionRead)
	for _, menu := range options {
		childrenMap[menu.ParentID] = append(childrenMap[menu.ParentID], menu)
	}

	result := make([]vo.Option, 0)
	for _, menu := range childrenMap[0] {
		result = append(result, buildOptionVO(menu, childrenMap))
	}
	return result, nil
}

func (s *MenuService) GetRoutes(ctx context.Context, roles []string) ([]vo.RouteVO, error) {
	// 尝试从缓存获取路由列表
	if s.cache != nil {
		cachedData, err := s.cache.Get(ctx, ROUTE_CACHE_KEY)
		if err == nil && cachedData != "" {
			// 缓存命中，反序列化返回
			var cachedRoutes []vo.RouteVO
			if unmarshalErr := json.Unmarshal([]byte(cachedData), &cachedRoutes); unmarshalErr == nil {
				return cachedRoutes, nil
			}
		}
	}

	// 缓存未命中，从数据库查询
	routes, err := s.menuRepo.GetMenuRoutes(ctx, roles)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询路由列表失败", err)
	}

	result := buildRoutesVO(routes)

	// 写入缓存
	if s.cache != nil {
		if data, marshalErr := json.Marshal(result); marshalErr == nil {
			_ = s.cache.Set(ctx, ROUTE_CACHE_KEY, string(data), ROUTE_CACHE_EXPIRATION)
		}
	}

	return result, nil
}

func (s *MenuService) clearAllRolePermsCache(ctx context.Context) {
	if s.cache == nil {
		return
	}
	// 角色权限缓存采用逐角色独立 Key（role:perms:{code}），无法通过单次 Delete 清除全部。
	// Go 端鉴权使用 JWT（不读取此缓存），菜单变更后各角色权限 Key 将在 TTL（30min）内自然过期。
	// 如需立即失效，应由 RoleService.refreshRolePermsCache 逐角色刷新。
	// 此处仅清除路由缓存。
	_ = s.cache.Delete(ctx, ROUTE_CACHE_KEY)
}

func (s *MenuService) generateMenuTreePath(ctx context.Context, parentId int64) string {
	if parentId == 0 {
		return "0"
	}
	parent, err := s.menuRepo.FindByID(ctx, parentId)
	if err != nil || parent == nil {
		return "0"
	}
	return parent.TreePath + "," + strconv.FormatInt(parent.ID, 10)
}

// buildMenuVO 递归构建菜单 VO（使用 map 索引，O(n) 复杂度）
func buildMenuVO(menu model.SysMenu, childrenMap map[int64][]model.SysMenu) vo.MenuVO {
	menuVO := vo.MenuVO{
		ID:        menu.ID,
		ParentID:  menu.ParentID,
		Name:      menu.Name,
		Type:      enum.GetMenuTypeEnumName(int(menu.Type)),
		Path:      menu.Path,
		Component: menu.Component,
		Sort:      menu.Sort,
		Visible:   int(menu.Visible),
		Icon:      menu.Icon,
		Redirect:  menu.Redirect,
		Perm:      menu.Perm,
		Children:  []vo.MenuVO{},
	}
	for _, child := range childrenMap[menu.ID] {
		menuVO.Children = append(menuVO.Children, buildMenuVO(child, childrenMap))
	}
	return menuVO
}

// buildOptionVO 递归构建菜单选项 VO（使用 map 索引，O(n) 复杂度）
func buildOptionVO(menu read.MenuOptionRead, childrenMap map[int64][]read.MenuOptionRead) vo.Option {
	option := vo.Option{
		Value: menu.ID,
		Label: menu.Name,
	}
	children := make([]vo.Option, 0)
	for _, child := range childrenMap[menu.ID] {
		children = append(children, buildOptionVO(child, childrenMap))
	}
	if len(children) > 0 {
		option.Children = children
	}
	return option
}

// buildRoutesVO 构建路由树形列表（使用 map 索引，O(n) 复杂度）
func buildRoutesVO(routeList []read.MenuRouteRead) []vo.RouteVO {
	// 按 ParentID 分组
	childrenMap := make(map[int64][]read.MenuRouteRead)
	for _, route := range routeList {
		childrenMap[route.ParentID] = append(childrenMap[route.ParentID], route)
	}

	routes := make([]vo.RouteVO, 0)
	for _, route := range childrenMap[0] {
		routes = append(routes, buildRouteVO(route, childrenMap))
	}
	return routes
}

// buildRouteVO 递归构建路由 VO
func buildRouteVO(route read.MenuRouteRead, childrenMap map[int64][]read.MenuRouteRead) vo.RouteVO {
	// 处理角色列表：将逗号分隔的字符串解析为切片
	var roles []string
	if route.Roles != "" {
		roles = strings.Split(route.Roles, ",")
	} else {
		roles = []string{}
	}

	meta := vo.RouteMeta{
		Title:  route.Name,
		Icon:   route.Icon,
		Roles:  roles,
		Hidden: route.Visible == 0,
	}

	if route.Type == enum.MenuTypeCatalog && route.KeepAlive == 1 {
		keepAlive := true
		meta.KeepAlive = &keepAlive
	}

	if route.Type == enum.MenuTypeMenu && route.AlwaysShow == 1 {
		alwaysShow := true
		meta.AlwaysShow = &alwaysShow
	}

	routeVO := vo.RouteVO{
		Name:      utils.ToCamelCase(route.Path),
		Path:      route.Path,
		Redirect:  route.Redirect,
		Component: route.Component,
		Meta:      meta,
	}

	children := make([]vo.RouteVO, 0)
	for _, child := range childrenMap[route.ID] {
		children = append(children, buildRouteVO(child, childrenMap))
	}
	if len(children) > 0 {
		routeVO.Children = children
	}

	return routeVO
}

// UpdateMenuVisible 修改菜单显示状态（额外方法，不在 IMenuService 接口中）
func (s *MenuService) UpdateMenuVisible(ctx context.Context, menuId int64, visible int) error {
	menu, err := s.menuRepo.FindByID(ctx, menuId)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询菜单失败", err)
	}
	if menu == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "菜单不存在")
	}

	menu.Visible = int8(visible)
	if err := s.menuRepo.Update(ctx, menu); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新菜单显示状态失败", err)
	}

	// 刷新权限缓存
	s.clearAllRolePermsCache(ctx)
	return nil
}

// validateMenuForm 校验菜单表单数据
// excludeID: 排除的菜单ID（用于更新时排除自身）
func (s *MenuService) validateMenuForm(ctx context.Context, form *bo.MenuForm, excludeID int64) error {
	// 1. 父菜单存在性校验
	if form.ParentID > 0 {
		parent, err := s.menuRepo.FindByID(ctx, form.ParentID)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询父菜单失败", err)
		}
		if parent == nil {
			return common.NewBizError(common.PARAM_ERROR, "父菜单不存在")
		}

		// 1.1 上级菜单类型校验：父菜单不能是按钮类型
		if parent.Type == enum.MenuTypeButton {
			return common.NewBizError(common.PARAM_ERROR, "父菜单不能是按钮类型")
		}

		// 1.2 上级菜单类型校验：父菜单不能是外链类型
		if parent.Type == enum.MenuTypeExtlink {
			return common.NewBizError(common.PARAM_ERROR, "父菜单不能是外链类型")
		}

		// 1.2 上级菜单类型校验：按钮只能挂在菜单类型下
		if form.Type == enum.MenuTypeButton && parent.Type != enum.MenuTypeMenu {
			return common.NewBizError(common.PARAM_ERROR, "按钮只能挂在菜单类型下")
		}

		// 1.3 层级限制校验：最多5层
		depth, err := s.getMenuDepth(ctx, form.ParentID)
		if err != nil {
			return err
		}
		if depth >= 5 {
			return common.NewBizError(common.PARAM_ERROR, "菜单层级不能超过5层")
		}

		// 1.4 循环引用校验：更新时不能将父菜单设置为自己或自己的子菜单
		if excludeID > 0 {
			isDesc, err := s.isDescendant(ctx, excludeID, form.ParentID)
			if err != nil {
				return err
			}
			if isDesc {
				return common.NewBizError(common.PARAM_ERROR, "不能将父菜单设置为自己或自己的子菜单")
			}
		}
	}

	// 2. 同级菜单名称唯一性校验
	exists, err := s.menuRepo.ExistsByName(ctx, form.ParentID, form.Name, excludeID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "校验菜单名称失败", err)
	}
	if exists {
		return common.NewBizError(common.PARAM_ERROR, "同级菜单名称已存在")
	}

	// 3. 同级菜单路径唯一性校验（菜单和目录类型需要校验）
	if form.Type != 4 && form.Path != "" {
		exists, err = s.menuRepo.ExistsByPath(ctx, form.ParentID, form.Path, excludeID)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "校验菜单路径失败", err)
		}
		if exists {
			return common.NewBizError(common.PARAM_ERROR, "同级菜单路径已存在")
		}
	}

	// 4. 权限标识全局唯一性校验
	if form.Perm != "" {
		exists, err = s.menuRepo.ExistsByPerm(ctx, form.Perm, excludeID)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "校验权限标识失败", err)
		}
		if exists {
			return common.NewBizError(common.PARAM_ERROR, "权限标识已存在")
		}
	}

	// 4. 菜单类型关联字段校验
	// 4.1 目录类型必须有路径
	if form.Type == enum.MenuTypeCatalog && form.Path == "" {
		return common.NewBizError(common.PARAM_ERROR, "目录类型必须配置路由路径")
	}
	// 4.2 菜单类型必须有路径
	if form.Type == enum.MenuTypeMenu && form.Path == "" {
		return common.NewBizError(common.PARAM_ERROR, "菜单类型必须配置路由路径")
	}
	// 4.3 外链类型必须有路径（外链地址）
	if form.Type == enum.MenuTypeExtlink && form.Path == "" {
		return common.NewBizError(common.PARAM_ERROR, "外链类型必须配置外链地址")
	}
	// 4.4 目录类型必须有组件（通常是Layout）
	if form.Type == enum.MenuTypeCatalog && form.Component == "" {
		return common.NewBizError(common.PARAM_ERROR, "目录类型必须配置组件")
	}
	// 4.5 菜单类型必须有组件
	if form.Type == enum.MenuTypeMenu && form.Component == "" {
		return common.NewBizError(common.PARAM_ERROR, "菜单类型必须配置组件")
	}

	return nil
}

// getMenuDepth 获取菜单深度（从根菜单到当前菜单的层级数）
func (s *MenuService) getMenuDepth(ctx context.Context, menuID int64) (int, error) {
	if menuID == 0 {
		return 0, nil
	}
	menu, err := s.menuRepo.FindByID(ctx, menuID)
	if err != nil {
		return 0, common.WrapBizError(common.DATABASE_ERROR, "查询菜单失败", err)
	}
	if menu == nil {
		return 0, nil
	}
	if menu.TreePath == "" {
		return 1, nil
	}
	// 通过TreePath计算深度，TreePath格式为 "0,1,2"，层级数 = 路径节点数
	return len(strings.Split(menu.TreePath, ",")), nil
}

// isDescendant 检查targetID是否是ancestorID的后代
func (s *MenuService) isDescendant(ctx context.Context, ancestorID int64, targetID int64) (bool, error) {
	if targetID == 0 {
		return false, nil
	}
	if ancestorID == targetID {
		return true, nil
	}
	// 获取targetID的菜单信息
	target, err := s.menuRepo.FindByID(ctx, targetID)
	if err != nil {
		return false, common.WrapBizError(common.DATABASE_ERROR, "查询菜单失败", err)
	}
	if target == nil {
		return false, nil
	}
	// 通过TreePath判断：如果TreePath包含ancestorID，则是后代
	if target.TreePath != "" {
		// TreePath格式为 "0,1,2"，检查是否包含ancestorID
		ancestorIDStr := strconv.FormatInt(ancestorID, 10)
		parts := strings.Split(target.TreePath, ",")
		for _, part := range parts {
			if part == ancestorIDStr {
				return true, nil
			}
		}
	}
	return false, nil
}

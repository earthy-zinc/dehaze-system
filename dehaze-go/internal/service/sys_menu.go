package service

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/global"
	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"gorm.io/gorm"
)

type MenuService struct {
	menuRepo repository.IMenuRepository
}

func NewMenuService(menuRepo repository.IMenuRepository) *MenuService {
	return &MenuService{menuRepo: menuRepo}
}

func (s *MenuService) getRepo() repository.IMenuRepository {
	if s.menuRepo != nil {
		return s.menuRepo
	}
	return repository.NewMenuRepository(global.DB)
}

func (s *MenuService) GetList(ctx context.Context, q *query.MenuQuery) ([]vo.MenuVO, error) {
	repo := s.getRepo()
	if q == nil {
		q = &query.MenuQuery{}
	}

	menus, err := repo.FindAll(ctx, q)
	if err != nil {
		return nil, err
	}

	return buildMenuTree(0, menus), nil
}

func (s *MenuService) GetFormData(ctx context.Context, id int64) (*bo.MenuForm, error) {
	repo := s.getRepo()
	form, err := repo.GetFormData(ctx, id)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, errors.New("菜单不存在")
		}
		return nil, err
	}
	if form == nil {
		return nil, errors.New("菜单不存在")
	}
	return form, nil
}

func (s *MenuService) Create(ctx context.Context, form *bo.MenuForm) error {
	if form == nil {
		return errors.New("表单数据不能为空")
	}

	menuType := form.Type
	path := form.Path

	if menuType == 2 {
		if form.ParentID == 0 && !strings.HasPrefix(path, "/") {
			path = "/" + path
		}
		form.Component = "Layout"
	} else if menuType == 3 {
		form.Component = ""
	}

	repo := s.getRepo()
	treePath := s.generateMenuTreePath(ctx, form.ParentID)

	menu := &model.SysMenu{
		ParentID:   form.ParentID,
		Name:       form.Name,
		Type:       form.Type,
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

	if err := repo.Create(ctx, menu); err != nil {
		return err
	}

	s.clearAllRolePermsCache()
	return nil
}

func (s *MenuService) Update(ctx context.Context, id int64, form *bo.MenuForm) error {
	if form == nil {
		return errors.New("表单数据不能为空")
	}

	menuType := form.Type
	path := form.Path

	if menuType == 2 {
		if form.ParentID == 0 && !strings.HasPrefix(path, "/") {
			path = "/" + path
		}
		form.Component = "Layout"
	} else if menuType == 3 {
		form.Component = ""
	}

	repo := s.getRepo()
	treePath := s.generateMenuTreePath(ctx, form.ParentID)

	menu := &model.SysMenu{
		BaseModel:  model.BaseModel{ID: id},
		ParentID:   form.ParentID,
		Name:       form.Name,
		Type:       form.Type,
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

	if err := repo.Update(ctx, menu); err != nil {
		return err
	}

	s.clearAllRolePermsCache()
	return nil
}

func (s *MenuService) Delete(ctx context.Context, id int64) error {
	repo := s.getRepo()

	if err := repo.Delete(ctx, id); err != nil {
		return err
	}

	s.clearAllRolePermsCache()
	return nil
}

func (s *MenuService) GetOptions(ctx context.Context) ([]vo.Option, error) {
	repo := s.getRepo()

	menuList, err := repo.FindAll(ctx, &query.MenuQuery{})
	if err != nil {
		return nil, err
	}

	options := buildMenuOptions(0, menuList)
	return options, nil
}

func (s *MenuService) GetRoutes(ctx context.Context, roles []string) ([]vo.RouteVO, error) {
	repo := s.getRepo()

	menus, err := repo.FindRoutesByRoles(ctx, roles)
	if err != nil {
		return nil, err
	}

	routeBOs := make([]bo.RouteBO, len(menus))
	for i, menu := range menus {
		routeBOs[i] = bo.RouteBO{
			ID:         menu.ID,
			ParentID:   menu.ParentID,
			Name:       menu.Name,
			Type:       menu.Type,
			Path:       menu.Path,
			Component:  menu.Component,
			Perm:       menu.Perm,
			Visible:    int(menu.Visible),
			Sort:       menu.Sort,
			Icon:       menu.Icon,
			Redirect:   menu.Redirect,
			Roles:      []string{},
			AlwaysShow: int(menu.AlwaysShow),
			KeepAlive:  int(menu.KeepAlive),
		}
	}

	return buildRoutes(0, routeBOs), nil
}

func (s *MenuService) clearAllRolePermsCache() {
	if global.REDIS == nil {
		return
	}
	ctx := context.Background()
	global.REDIS.Del(ctx, "role:perms")
}

func (s *MenuService) generateMenuTreePath(ctx context.Context, parentId int64) string {
	if parentId == 0 {
		return "0"
	}
	repo := s.getRepo()
	parent, err := repo.FindByID(ctx, parentId)
	if err != nil || parent == nil {
		return "0"
	}
	return parent.TreePath + "," + strconv.FormatInt(parent.ID, 10)
}

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
	if menuVOs == nil {
		return []vo.MenuVO{}
	}
	return menuVOs
}

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
	if options == nil {
		return []vo.Option{}
	}
	return options
}

func buildRoutes(parentId int64, routeList []bo.RouteBO) []vo.RouteVO {
	var routes []vo.RouteVO
	for _, route := range routeList {
		if route.ParentID == parentId {
			meta := vo.RouteMeta{
				Title:  route.Name,
				Icon:   route.Icon,
				Roles:  route.Roles,
				Hidden: route.Visible == 0,
			}

			if route.Type == 1 && route.KeepAlive == 1 {
				keepAlive := true
				meta.KeepAlive = &keepAlive
			}

			if route.Type == 2 && route.AlwaysShow == 1 {
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

			children := buildRoutes(route.ID, routeList)
			if len(children) > 0 {
				routeVO.Children = children
			}

			routes = append(routes, routeVO)
		}
	}
	if routes == nil {
		return []vo.RouteVO{}
	}
	return routes
}

// UpdateMenuVisible 修改菜单显示状态（额外方法，不在 IMenuService 接口中）
func (s *MenuService) UpdateMenuVisible(ctx context.Context, menuId int64, visible int) error {
	repo := s.getRepo()
	menu, err := repo.FindByID(ctx, menuId)
	if err != nil {
		return err
	}
	if menu == nil {
		return errors.New("菜单不存在")
	}

	menu.Visible = int8(visible)
	return repo.Update(ctx, menu)
}

package menu

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

// IMenuService 菜单服务接口
type IMenuService interface {
	// GetList 获取菜单列表
	GetList(ctx context.Context, q *query.MenuQuery) ([]vo.MenuVO, error)

	// GetFormData 获取菜单表单数据
	GetFormData(ctx context.Context, id int64) (*bo.MenuForm, error)

	// Create 创建菜单
	Create(ctx context.Context, form *bo.MenuForm) error

	// Update 更新菜单
	Update(ctx context.Context, id int64, form *bo.MenuForm) error

	// Delete 删除菜单
	Delete(ctx context.Context, id int64) error

	// GetOptions 获取菜单下拉选项
	GetOptions(ctx context.Context) ([]vo.Option, error)

	// GetRoutes 获取当前用户路由菜单
	GetRoutes(ctx context.Context, roles []string) ([]vo.RouteVO, error)
}

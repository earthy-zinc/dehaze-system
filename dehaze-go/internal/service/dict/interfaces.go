package dict

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

// ====================
// 字典管理服务接口
// ====================

// IDictTypeService 字典类型服务接口
type IDictTypeService interface {
	// GetPage 字典类型分页列表
	GetPage(ctx context.Context, q *query.DictTypePageQuery) (*vo.PageResult[vo.DictTypePageVO], error)

	// GetFormData 获取字典类型表单数据
	GetFormData(ctx context.Context, id int64) (*bo.DictTypeFormBO, error)

	// Create 创建字典类型
	Create(ctx context.Context, form *bo.DictTypeFormBO) error

	// Update 更新字典类型
	Update(ctx context.Context, id int64, form *bo.DictTypeFormBO) error

	// Delete 删除字典类型
	Delete(ctx context.Context, ids []int64) error
}

// IDictService 字典数据服务接口
type IDictService interface {
	// GetPage 字典数据分页列表
	GetPage(ctx context.Context, q *query.DictPageQuery) (*vo.PageResult[vo.DictPageVO], error)

	// GetByTypeCode 根据类型编码获取字典列表
	GetByTypeCode(ctx context.Context, typeCode string) ([]vo.Option, error)

	// GetFormData 获取字典表单数据
	GetFormData(ctx context.Context, id int64) (*bo.DictFormBO, error)

	// Create 创建字典
	Create(ctx context.Context, form *bo.DictFormBO) error

	// Update 更新字典
	Update(ctx context.Context, id int64, form *bo.DictFormBO) error

	// Delete 删除字典
	Delete(ctx context.Context, ids []int64) error
}

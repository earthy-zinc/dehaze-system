package dict

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
	"gorm.io/gorm"
)

// ====================
// 字典管理 Repository
// ====================

// IDictTypeRepository 字典类型仓储接口
type IDictTypeRepository interface {
	// FindByID 根据 ID 查询字典类型
	FindByID(ctx context.Context, id int64) (*model.SysDictType, error)

	// FindByCode 根据编码查询字典类型
	FindByCode(ctx context.Context, code string) (*model.SysDictType, error)

	// ExistsByCode 检查字典类型编码是否存在
	ExistsByCode(ctx context.Context, code string, excludeID ...int64) (bool, error)

	// FindPage 分页查询字典类型
	FindPage(ctx context.Context, q *query.DictTypePageQuery) (*read.PageResult[read.DictTypePage], error)

	// Create 创建字典类型
	Create(ctx context.Context, dictType *model.SysDictType) error

	// Update 更新字典类型
	Update(ctx context.Context, dictType *model.SysDictType) error

	// Delete 删除字典类型
	Delete(ctx context.Context, ids []int64) error

	// FindCodesByIDs 根据 ID 列表查询编码列表
	FindCodesByIDs(ctx context.Context, ids []int64) ([]string, error)

	// Transaction 执行事务
	Transaction(ctx context.Context, fn func(repo IDictTypeRepository) error) error

	// WithDB 返回使用指定 DB 的新实例（用于跨 Repository 事务）
	WithDB(db *gorm.DB) IDictTypeRepository
}

// IDictRepository 字典数据仓储接口
type IDictRepository interface {
	// FindByID 根据 ID 查询字典
	FindByID(ctx context.Context, id int64) (*model.SysDict, error)

	// FindByIDs 根据 ID 列表批量查询字典
	FindByIDs(ctx context.Context, ids []int64) ([]model.SysDict, error)

	// FindByTypeCode 根据类型编码查询字典列表
	FindByTypeCode(ctx context.Context, typeCode string) ([]model.SysDict, error)

	// FindByTypeCodes 根据类型编码列表查询字典列表
	FindByTypeCodes(ctx context.Context, typeCodes []string) ([]model.SysDict, error)

	// FindPage 分页查询字典
	FindPage(ctx context.Context, q *query.DictPageQuery) (*read.PageResult[read.DictPage], error)

	// Create 创建字典
	Create(ctx context.Context, dict *model.SysDict) error

	// Update 更新字典
	Update(ctx context.Context, dict *model.SysDict) error

	// UpdateTypeCode 批量更新字典的类型编码
	UpdateTypeCode(ctx context.Context, oldCode, newCode string) error

	// Delete 删除字典
	Delete(ctx context.Context, ids []int64) error

	// DeleteByTypeCodes 根据类型编码列表删除字典
	DeleteByTypeCodes(ctx context.Context, typeCodes []string) error

	// CountByTypeCodes 根据类型编码列表统计字典数量
	CountByTypeCodes(ctx context.Context, typeCodes []string) (int64, error)

	// ExistsByTypeCodeAndValue 检查同一类型下字典值是否存在
	ExistsByTypeCodeAndValue(ctx context.Context, typeCode, value string, excludeID ...int64) (bool, error)

	// Transaction 执行事务
	Transaction(ctx context.Context, fn func(repo IDictRepository) error) error

	// WithDB 返回使用指定 DB 的新实例（用于跨 Repository 事务）
	WithDB(db *gorm.DB) IDictRepository
}

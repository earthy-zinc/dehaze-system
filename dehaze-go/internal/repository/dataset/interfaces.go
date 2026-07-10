package dataset

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
)

// ====================
// 数据集管理 Repository
// ====================

// IDatasetRepository 数据集仓储接口
type IDatasetRepository interface {
	// FindByID 根据 ID 查询数据集
	FindByID(ctx context.Context, id int64) (*model.SysDataset, error)

	// FindAll 查询所有数据集
	FindAll(ctx context.Context) ([]model.SysDataset, error)

	// FindAllActive 查询所有活跃数据集
	FindAllActive(ctx context.Context) ([]model.SysDataset, error)

	// FindPage 分页查询数据集
	FindPage(ctx context.Context, q *query.DatasetQuery) (*read.PageResult[read.Dataset], error)

	// ExistsByParentIDAndName 检查同一父数据集下是否存在同名数据集
	ExistsByParentIDAndName(ctx context.Context, parentID int64, name string, excludeID int64) (bool, error)

	// Create 创建数据集
	Create(ctx context.Context, dataset *model.SysDataset) error

	// Update 更新数据集
	Update(ctx context.Context, dataset *model.SysDataset) error

	// Delete 删除数据集
	Delete(ctx context.Context, ids []int64) error

	// SoftDeleteByIDs 批量逻辑删除数据集
	SoftDeleteByIDs(ctx context.Context, ids []int64, updateBy int64) error

	// GetFormData 获取数据集表单数据
	GetFormData(ctx context.Context, datasetID int64) (*read.DatasetForm, error)

	// Transaction 执行事务
	Transaction(ctx context.Context, fn func(txRepo IDatasetRepository) error) error
}

// IDatasetItemRepository 数据项仓储接口
type IDatasetItemRepository interface {
	// FindByID 根据 ID 查询数据项
	FindByID(ctx context.Context, id int64) (*model.SysDatasetItem, error)

	// FindByDatasetID 根据数据集 ID 查询数据项
	FindByDatasetID(ctx context.Context, datasetID int64) ([]model.SysDatasetItem, error)

	// FindByDatasetIDs 根据数据集 ID 列表查询数据项
	FindByDatasetIDs(ctx context.Context, datasetIDs []int64) ([]model.SysDatasetItem, error)

	// CountByDatasetIDs 根据数据集 ID 列表统计数据项数量
	CountByDatasetIDs(ctx context.Context, datasetIDs []int64) (int64, error)

	// FindIDsByDatasetIDs 根据数据集 ID 列表查询数据项 ID 列表
	FindIDsByDatasetIDs(ctx context.Context, datasetIDs []int64) ([]int64, error)

	// Create 创建数据项
	Create(ctx context.Context, item *model.SysDatasetItem) error

	// BatchCreate 批量创建数据项
	BatchCreate(ctx context.Context, items []model.SysDatasetItem) error

	// Delete 删除数据项
	Delete(ctx context.Context, ids []int64) error

	// DeleteByDatasetID 根据数据集 ID 删除数据项
	DeleteByDatasetID(ctx context.Context, datasetID int64) error

	// DeleteByDatasetIDs 根据数据集 ID 列表删除数据项
	DeleteByDatasetIDs(ctx context.Context, datasetIDs []int64) error

	// FindPage 分页查询数据项
	FindPage(ctx context.Context, datasetID int64, pageNum, pageSize int) ([]model.SysDatasetItem, int64, error)

	// Update 更新数据项
	Update(ctx context.Context, item *model.SysDatasetItem) error
}

// IDatasetItemFileRepository 数据项和项文件联合操作接口（用于事务）
type IDatasetItemFileRepository interface {
	// CreateDatasetItemWithFiles 创建数据项及其关联文件（事务）
	CreateDatasetItemWithFiles(ctx context.Context, datasetID int64, itemName string, files []ItemFileCreate) (int64, []int64, error)

	// DeleteDatasetItemCascade 级联删除数据项及其文件（事务）
	DeleteDatasetItemCascade(ctx context.Context, itemID int64) error

	// DeleteDatasetItemsCascade 批量级联删除数据项及其文件（事务）
	DeleteDatasetItemsCascade(ctx context.Context, itemIDs []int64) error
}

// ItemFileCreate 创建项文件参数
type ItemFileCreate struct {
	Type      string
	Name      string
	Path      string
	URL       string
	Size      int64
	MD5       string
	HazeLevel string
	FileType  string
}

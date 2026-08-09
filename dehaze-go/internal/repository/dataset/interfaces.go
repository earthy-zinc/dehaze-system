package dataset

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
)

const ROOT_NODE_ID = 0

type CountByDatasetResult struct {
	DatasetID int64 `gorm:"column:dataset_id"`
	Cnt       int64 `gorm:"column:cnt"`
}

type DatasetStatsResult struct {
	DatasetID        int64 `gorm:"column:dataset_id"`
	FileCount        int64 `gorm:"column:image_count"`
	TotalSize        int64 `gorm:"column:total_size"`
	AnnotatedCount   int64 `gorm:"column:annotated_count"`
	UnannotatedCount int64 `gorm:"column:unannotated_count"`
}

type DistributionResult struct {
	DatasetID int64  `gorm:"column:dataset_id"`
	Key       string `gorm:"column:dist_key"`
	Cnt       int64  `gorm:"column:cnt"`
}

type IDatasetRepository interface {
	FindByID(ctx context.Context, id int64) (*model.SysDataset, error)
	FindAll(ctx context.Context) ([]model.SysDataset, error)
	FindAllActive(ctx context.Context) ([]model.SysDataset, error)
	FindRootPage(ctx context.Context, q *query.DatasetQuery) ([]model.SysDataset, int64, error)
	FindByParentID(ctx context.Context, parentID int64) ([]model.SysDataset, error)
	FindByParentIDs(ctx context.Context, parentIDs []int64) ([]model.SysDataset, error)
	CountHasChildren(ctx context.Context, parentIDs []int64) (map[int64]bool, error)
	ExistsByParentIDAndName(ctx context.Context, parentID int64, name string, excludeID int64) (bool, error)
	ExistsByID(ctx context.Context, id int64) (bool, error)
	Create(ctx context.Context, dataset *model.SysDataset) error
	Update(ctx context.Context, dataset *model.SysDataset) error
	SoftDeleteByIDs(ctx context.Context, ids []int64, updateBy int64) error
	GetFormData(ctx context.Context, datasetID int64) (*model.SysDataset, error)
	Transaction(ctx context.Context, fn func(txRepo IDatasetRepository) error) error
}

type IDatasetItemRepository interface {
	FindByID(ctx context.Context, id int64) (*model.SysDatasetItem, error)
	FindByDatasetID(ctx context.Context, datasetID int64) ([]model.SysDatasetItem, error)
	FindByDatasetIDs(ctx context.Context, datasetIDs []int64) ([]model.SysDatasetItem, error)
	CountByDatasetIDs(ctx context.Context, datasetIDs []int64) (int64, error)
	FindIDsByDatasetIDs(ctx context.Context, datasetIDs []int64) ([]int64, error)
	CountItemsPerDataset(ctx context.Context, datasetIDs []int64) ([]CountByDatasetResult, error)
	Create(ctx context.Context, item *model.SysDatasetItem) error
	BatchCreate(ctx context.Context, items []model.SysDatasetItem) error
	Delete(ctx context.Context, ids []int64) error
	DeleteByDatasetID(ctx context.Context, datasetID int64) error
	DeleteByDatasetIDs(ctx context.Context, datasetIDs []int64) error
	FindPage(ctx context.Context, datasetID int64, pageNum, pageSize int) ([]model.SysDatasetItem, int64, error)
	Update(ctx context.Context, item *model.SysDatasetItem) error
}

type IDatasetStatsRepository interface {
	CountDatasetStatsBatch(ctx context.Context, datasetIDs []int64) ([]DatasetStatsResult, error)
	CountSceneDistributionBatch(ctx context.Context, datasetIDs []int64) ([]DistributionResult, error)
	CountHazeDistributionBatch(ctx context.Context, datasetIDs []int64) ([]DistributionResult, error)
	CountFormatDistributionBatch(ctx context.Context, datasetIDs []int64) ([]DistributionResult, error)
}

type IDatasetItemFileRepository interface {
	CreateDatasetItemWithFiles(ctx context.Context, datasetID int64, itemName string, files []ItemFileCreate) (int64, []int64, error)
	DeleteDatasetItemCascade(ctx context.Context, itemID int64) error
	DeleteDatasetItemsCascade(ctx context.Context, itemIDs []int64) error
}

type ItemFileCreate struct {
	Type       string
	Name       string
	ObjectName string
	Storage    string
	Size       int64
	MD5        string
	HazeLevel  string
	FileType   string
	SceneType  string
}

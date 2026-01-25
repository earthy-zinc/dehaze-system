package repository

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"

	"gorm.io/gorm"
)

// DatasetRepository 数据集仓储实现
type DatasetRepository struct {
	db *gorm.DB
}

// NewDatasetRepository 创建数据集仓储实例
func NewDatasetRepository(db *gorm.DB) *DatasetRepository {
	return &DatasetRepository{db: db}
}

// FindByID 根据 ID 查询数据集
func (r *DatasetRepository) FindByID(ctx context.Context, id int64) (*model.SysDataset, error) {
	var dataset model.SysDataset
	err := r.db.WithContext(ctx).
		Where("id = ? AND deleted = ?", id, 0).
		First(&dataset).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &dataset, err
}

// FindPage 分页查询数据集
func (r *DatasetRepository) FindPage(ctx context.Context, q *query.DatasetQuery) (*vo.PageResult[vo.DatasetVO], error) {
	db := r.db.WithContext(ctx).Model(&model.SysDataset{}).
		Where("deleted = ?", 0)

	if q != nil && q.Keywords != "" {
		keyword := "%" + q.Keywords + "%"
		db = db.Where("name LIKE ?", keyword)
	}

	var datasetList []model.SysDataset
	err := db.Find(&datasetList).Error
	if err != nil {
		return nil, err
	}

	datasetVOs := r.buildDatasetTree(datasetList, 0)
	return &vo.PageResult[vo.DatasetVO]{
		List:  datasetVOs,
		Total: int64(len(datasetList)),
	}, nil
}

// buildDatasetTree 构建数据集树形结构
func (r *DatasetRepository) buildDatasetTree(datasetList []model.SysDataset, rootID int64) []vo.DatasetVO {
	if len(datasetList) == 0 {
		return []vo.DatasetVO{}
	}

	datasetMap := make(map[int64]model.SysDataset)
	for _, dataset := range datasetList {
		datasetMap[dataset.ID] = dataset
	}

	parentToChildren := make(map[int64][]model.SysDataset)
	for _, dataset := range datasetList {
		parentID := dataset.ParentID
		if parentToChildren[parentID] == nil {
			parentToChildren[parentID] = []model.SysDataset{}
		}
		parentToChildren[parentID] = append(parentToChildren[parentID], dataset)
	}

	var roots []model.SysDataset
	if rootID != 0 {
		if children, ok := parentToChildren[rootID]; ok {
			roots = children
		}
	} else {
		rootIDs := r.findRootIDs(datasetList)
		for _, rid := range rootIDs {
			if children, ok := parentToChildren[rid]; ok {
				roots = append(roots, children...)
			}
		}
	}

	result := make([]vo.DatasetVO, 0, len(roots))
	for _, root := range roots {
		result = append(result, r.buildNodeTree(root, parentToChildren)...)
	}

	return result
}

// findRootIDs 查找根节点 ID
func (r *DatasetRepository) findRootIDs(datasetList []model.SysDataset) []int64 {
	idSet := make(map[int64]bool)
	for _, dataset := range datasetList {
		idSet[dataset.ID] = true
	}

	rootIDs := make([]int64, 0)
	for _, dataset := range datasetList {
		if !idSet[dataset.ParentID] {
			rootIDs = append(rootIDs, dataset.ParentID)
		}
	}
	return rootIDs
}

// buildNodeTree 递归构建节点树
func (r *DatasetRepository) buildNodeTree(dataset model.SysDataset, parentToChildren map[int64][]model.SysDataset) []vo.DatasetVO {
	datasetVO := vo.DatasetVO{
		ID:          dataset.ID,
		ParentID:    dataset.ParentID,
		Type:        dataset.Type,
		Name:        dataset.Name,
		Description: dataset.Description,
		Path:        dataset.Path,
		Size:        dataset.Size,
		CreateTime:  dataset.CreatedAt,
		UpdateTime:  dataset.UpdatedAt,
		Status:      int(dataset.Status),
	}

	if children, ok := parentToChildren[dataset.ID]; ok {
		datasetVO.Children = make([]vo.DatasetVO, 0, len(children))
		for _, child := range children {
			datasetVO.Children = append(datasetVO.Children, r.buildNodeTree(child, parentToChildren)...)
		}
	}

	return []vo.DatasetVO{datasetVO}
}

// Create 创建数据集
func (r *DatasetRepository) Create(ctx context.Context, dataset *model.SysDataset) error {
	return r.db.WithContext(ctx).Create(dataset).Error
}

// Update 更新数据集
func (r *DatasetRepository) Update(ctx context.Context, dataset *model.SysDataset) error {
	return r.db.WithContext(ctx).Model(dataset).
		Select("parent_id", "type", "name", "description", "path", "status", "update_time").
		Updates(dataset).Error
}

// Delete 删除数据集
func (r *DatasetRepository) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Model(&model.SysDataset{}).
		Where("id IN ?", ids).
		Update("deleted", 1).Error
}

// GetFormData 获取数据集表单数据
func (r *DatasetRepository) GetFormData(ctx context.Context, datasetID int64) (*bo.DatasetFormBO, error) {
	var dataset model.SysDataset
	err := r.db.WithContext(ctx).
		Where("id = ? AND deleted = ?", datasetID, 0).
		First(&dataset).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return &bo.DatasetFormBO{}, nil
		}
		return nil, err
	}

	idPtr := dataset.ID
	form := &bo.DatasetFormBO{
		ID:          &idPtr,
		ParentID:    dataset.ParentID,
		Type:        dataset.Type,
		Name:        dataset.Name,
		Description: dataset.Description,
		Path:        dataset.Path,
		Status:      dataset.Status,
		CreateTime:  dataset.CreatedAt.Format("2006-01-02T15:04:05"),
		UpdateTime:  dataset.UpdatedAt.Format("2006-01-02T15:04:05"),
		Statistics: &bo.StatisticsBO{
			ItemCount:          0,
			FileCount:          0,
			TotalSize:          0,
			ClearCount:         0,
			HazyCount:          0,
			SceneDistribution:  make(map[string]int64),
			HazeDistribution:   make(map[string]int64),
			FormatDistribution: make(map[string]int64),
		},
	}
	return form, nil
}

// Ensure DatasetRepository implements IDatasetRepository
var _ IDatasetRepository = (*DatasetRepository)(nil)

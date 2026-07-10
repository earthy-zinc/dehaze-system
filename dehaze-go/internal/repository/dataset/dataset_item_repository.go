package dataset

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"

	"gorm.io/gorm"
)

// DatasetItemRepository 数据项仓储实现
type DatasetItemRepository struct {
	db *gorm.DB
}

// NewDatasetItemRepository 创建数据项仓储实例
func NewDatasetItemRepository(db *gorm.DB) *DatasetItemRepository {
	return &DatasetItemRepository{db: db}
}

// FindByID 根据 ID 查询数据项
func (r *DatasetItemRepository) FindByID(ctx context.Context, id int64) (*model.SysDatasetItem, error) {
	var item model.SysDatasetItem
	err := r.db.WithContext(ctx).
		Where("id = ?", id).
		First(&item).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &item, err
}

// FindByDatasetID 根据数据集 ID 查询数据项
func (r *DatasetItemRepository) FindByDatasetID(ctx context.Context, datasetID int64) ([]model.SysDatasetItem, error) {
	var items []model.SysDatasetItem
	err := r.db.WithContext(ctx).
		Where("dataset_id = ?", datasetID).
		Find(&items).Error
	return items, err
}

// Create 创建数据项
func (r *DatasetItemRepository) Create(ctx context.Context, item *model.SysDatasetItem) error {
	if item.CreatedAt.IsZero() {
		item.CreatedAt = time.Now()
	}
	if item.UpdatedAt.IsZero() {
		item.UpdatedAt = time.Now()
	}
	return r.db.WithContext(ctx).Create(item).Error
}

// BatchCreate 批量创建数据项
func (r *DatasetItemRepository) BatchCreate(ctx context.Context, items []model.SysDatasetItem) error {
	if len(items) == 0 {
		return nil
	}
	now := time.Now()
	for i := range items {
		if items[i].CreatedAt.IsZero() {
			items[i].CreatedAt = now
		}
		if items[i].UpdatedAt.IsZero() {
			items[i].UpdatedAt = now
		}
	}
	return r.db.WithContext(ctx).Create(&items).Error
}

// Delete 删除数据项
func (r *DatasetItemRepository) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Delete(&model.SysDatasetItem{}, ids).Error
}

// DeleteByDatasetID 根据数据集 ID 删除数据项
func (r *DatasetItemRepository) DeleteByDatasetID(ctx context.Context, datasetID int64) error {
	return r.db.WithContext(ctx).
		Where("dataset_id = ?", datasetID).
		Delete(&model.SysDatasetItem{}).Error
}

// FindPage 分页查询数据项
func (r *DatasetItemRepository) FindPage(ctx context.Context, datasetID int64, pageNum, pageSize int) ([]model.SysDatasetItem, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysDatasetItem{})

	if datasetID > 0 {
		db = db.Where("dataset_id = ?", datasetID)
	}

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var items []model.SysDatasetItem
	offset := (pageNum - 1) * pageSize
	err := db.Offset(offset).Limit(pageSize).Order("id DESC").Find(&items).Error
	return items, total, err
}

// Update 更新数据项
func (r *DatasetItemRepository) Update(ctx context.Context, item *model.SysDatasetItem) error {
	item.UpdatedAt = time.Now()
	return r.db.WithContext(ctx).Model(item).
		Select("name", "update_time").
		Updates(item).Error
}

// CountByDatasetIDs 根据数据集 ID 列表统计数据项数量
func (r *DatasetItemRepository) CountByDatasetIDs(ctx context.Context, datasetIDs []int64) (int64, error) {
	if len(datasetIDs) == 0 {
		return 0, nil
	}
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysDatasetItem{}).
		Where("dataset_id IN ?", datasetIDs).
		Count(&count).Error
	return count, err
}

// FindIDsByDatasetIDs 根据数据集 ID 列表查询数据项 ID 列表
func (r *DatasetItemRepository) FindIDsByDatasetIDs(ctx context.Context, datasetIDs []int64) ([]int64, error) {
	if len(datasetIDs) == 0 {
		return nil, nil
	}
	var ids []int64
	err := r.db.WithContext(ctx).
		Model(&model.SysDatasetItem{}).
		Where("dataset_id IN ?", datasetIDs).
		Pluck("id", &ids).Error
	return ids, err
}

// FindByDatasetIDs 根据数据集 ID 列表查询数据项
func (r *DatasetItemRepository) FindByDatasetIDs(ctx context.Context, datasetIDs []int64) ([]model.SysDatasetItem, error) {
	if len(datasetIDs) == 0 {
		return nil, nil
	}
	var items []model.SysDatasetItem
	err := r.db.WithContext(ctx).
		Where("dataset_id IN ?", datasetIDs).
		Find(&items).Error
	return items, err
}

// DeleteByDatasetIDs 根据数据集 ID 列表删除数据项
func (r *DatasetItemRepository) DeleteByDatasetIDs(ctx context.Context, datasetIDs []int64) error {
	if len(datasetIDs) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).
		Where("dataset_id IN ?", datasetIDs).
		Delete(&model.SysDatasetItem{}).Error
}

// Ensure DatasetItemRepository implements IDatasetItemRepository
var _ IDatasetItemRepository = (*DatasetItemRepository)(nil)

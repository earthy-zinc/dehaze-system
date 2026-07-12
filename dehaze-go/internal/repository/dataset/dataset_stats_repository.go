package dataset

import (
	"context"

	"gorm.io/gorm"
)

type DatasetStatsRepository struct {
	db *gorm.DB
}

func NewDatasetStatsRepository(db *gorm.DB) *DatasetStatsRepository {
	return &DatasetStatsRepository{db: db}
}

func (r *DatasetStatsRepository) CountDatasetStatsBatch(ctx context.Context, datasetIDs []int64) ([]DatasetStatsResult, error) {
	if len(datasetIDs) == 0 {
		return nil, nil
	}

	var results []DatasetStatsResult
	err := r.db.WithContext(ctx).Table("sys_dataset_item sdi").
		Select(`
			sdi.dataset_id AS dataset_id,
			COUNT(sif.id) AS image_count,
			COALESCE(SUM(CAST(sf.size AS UNSIGNED)), 0) AS total_size,
			SUM(CASE WHEN sif.haze_level IS NOT NULL AND sif.haze_level != '' THEN 1 ELSE 0 END) AS annotated_count,
			SUM(CASE WHEN sif.haze_level IS NULL OR sif.haze_level = '' THEN 1 ELSE 0 END) AS unannotated_count
		`).
		Joins("LEFT JOIN sys_item_file sif ON sif.item_id = sdi.id").
		Joins("LEFT JOIN sys_file sf ON sif.file_id = sf.id").
		Where("sdi.dataset_id IN ?", datasetIDs).
		Group("sdi.dataset_id").
		Scan(&results).Error

	return results, err
}

func (r *DatasetStatsRepository) CountSceneDistributionBatch(ctx context.Context, datasetIDs []int64) ([]DistributionResult, error) {
	if len(datasetIDs) == 0 {
		return nil, nil
	}

	var results []DistributionResult
	err := r.db.WithContext(ctx).Table("sys_item_file sif").
		Select(`
			sdi.dataset_id AS dataset_id,
			COALESCE(NULLIF(sif.scene_type, ''), '未分类') AS key,
			COUNT(*) AS cnt
		`).
		Joins("JOIN sys_dataset_item sdi ON sif.item_id = sdi.id").
		Where("sdi.dataset_id IN ?", datasetIDs).
		Group("sdi.dataset_id, COALESCE(NULLIF(sif.scene_type, ''), '未分类')").
		Scan(&results).Error

	return results, err
}

func (r *DatasetStatsRepository) CountHazeDistributionBatch(ctx context.Context, datasetIDs []int64) ([]DistributionResult, error) {
	if len(datasetIDs) == 0 {
		return nil, nil
	}

	var results []DistributionResult
	err := r.db.WithContext(ctx).Table("sys_item_file sif").
		Select(`
			sdi.dataset_id AS dataset_id,
			COALESCE(NULLIF(sif.haze_level, ''), '未标注') AS key,
			COUNT(*) AS cnt
		`).
		Joins("JOIN sys_dataset_item sdi ON sif.item_id = sdi.id").
		Where("sdi.dataset_id IN ?", datasetIDs).
		Group("sdi.dataset_id, COALESCE(NULLIF(sif.haze_level, ''), '未标注')").
		Scan(&results).Error

	return results, err
}

func (r *DatasetStatsRepository) CountFormatDistributionBatch(ctx context.Context, datasetIDs []int64) ([]DistributionResult, error) {
	if len(datasetIDs) == 0 {
		return nil, nil
	}

	var results []DistributionResult
	err := r.db.WithContext(ctx).Table("sys_item_file sif").
		Select(`
			sdi.dataset_id AS dataset_id,
			COALESCE(sf.type, '未知') AS key,
			COUNT(*) AS cnt
		`).
		Joins("JOIN sys_dataset_item sdi ON sif.item_id = sdi.id").
		Joins("JOIN sys_file sf ON sif.file_id = sf.id").
		Where("sdi.dataset_id IN ?", datasetIDs).
		Group("sdi.dataset_id, COALESCE(sf.type, '未知')").
		Scan(&results).Error

	return results, err
}

var _ IDatasetStatsRepository = (*DatasetStatsRepository)(nil)

package dataset

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"gorm.io/gorm"
)

// DatasetItemFileRepository 数据项与文件联合操作仓储实现
// 负责跨表事务操作（dataset_item / file / item_file）
type DatasetItemFileRepository struct {
	db *gorm.DB
}

// NewDatasetItemFileRepository 创建数据项与文件联合操作仓储实例
func NewDatasetItemFileRepository(db *gorm.DB) *DatasetItemFileRepository {
	return &DatasetItemFileRepository{db: db}
}

// CreateDatasetItemWithFiles 创建数据项及其关联文件（事务）
func (r *DatasetItemFileRepository) CreateDatasetItemWithFiles(ctx context.Context, datasetID int64, itemName string, files []ItemFileCreate) (int64, []int64, error) {
	var itemID int64
	fileIDs := make([]int64, 0, len(files))

	err := r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		item := model.SysDatasetItem{
			DatasetID: datasetID,
			Name:      itemName,
		}
		if err := tx.Create(&item).Error; err != nil {
			return fmt.Errorf("创建数据项失败: %w", err)
		}

		itemID = item.ID

		for _, file := range files {
			fileID, err := r.createOrReuseFile(ctx, tx, file)
			if err != nil {
				return err
			}
			fileIDs = append(fileIDs, fileID)

			itemFile := model.SysItemFile{
				ItemID: itemID,
				FileID: fileID,
				Type:   file.Type,
			}
			if file.HazeLevel != "" {
				hazeDesc := fmt.Sprintf("雾霾程度: %s", file.HazeLevel)
				itemFile.Description = &hazeDesc
			}
			if err := tx.Create(&itemFile).Error; err != nil {
				return fmt.Errorf("创建项文件关联失败: %w", err)
			}
		}
		return nil
	})

	if err != nil {
		return 0, nil, err
	}
	return itemID, fileIDs, nil
}

// DeleteDatasetItemCascade 级联删除数据项及其文件关联记录（事务）
func (r *DatasetItemFileRepository) DeleteDatasetItemCascade(ctx context.Context, itemID int64) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err := tx.Where("item_id = ?", itemID).Delete(&model.SysItemFile{}).Error; err != nil {
			return fmt.Errorf("删除项文件关联失败: %w", err)
		}
		if err := tx.Delete(&model.SysDatasetItem{}, itemID).Error; err != nil {
			return fmt.Errorf("删除数据项失败: %w", err)
		}
		return nil
	})
}

// DeleteDatasetItemsCascade 批量级联删除数据项及其文件关联记录（事务）
func (r *DatasetItemFileRepository) DeleteDatasetItemsCascade(ctx context.Context, itemIDs []int64) error {
	if len(itemIDs) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err := tx.Where("item_id IN ?", itemIDs).Delete(&model.SysItemFile{}).Error; err != nil {
			return fmt.Errorf("删除项文件关联失败: %w", err)
		}
		if err := tx.Where("id IN ?", itemIDs).Delete(&model.SysDatasetItem{}).Error; err != nil {
			return fmt.Errorf("删除数据项失败: %w", err)
		}
		return nil
	})
}

func (r *DatasetItemFileRepository) createOrReuseFile(ctx context.Context, tx *gorm.DB, file ItemFileCreate) (int64, error) {
	var existing model.SysFile
	if file.MD5 != "" {
		err := tx.WithContext(ctx).Where("md5 = ?", file.MD5).First(&existing).Error
		if err == nil {
			return int64(existing.ID), nil
		}
		if !errors.Is(err, gorm.ErrRecordNotFound) {
			return 0, fmt.Errorf("查询文件记录失败: %w", err)
		}
	}

	fileType := file.FileType
	if fileType == "" {
		fileType = getFileExtension(file.Name)
		if fileType == "" {
			fileType = getFileExtension(file.Path)
		}
	}

	var fileTypePtr *string
	if fileType != "" {
		fileTypePtr = utils.StringPtr(fileType)
	}

	var urlPtr *string
	if file.URL != "" {
		urlPtr = utils.StringPtr(file.URL)
	}

	newFile := model.SysFile{
		Type:       fileTypePtr,
		URL:        urlPtr,
		Name:       file.Name,
		ObjectName: file.Name,
		Path:       file.Path,
		Size:       fmt.Sprintf("%d", file.Size),
		MD5:        file.MD5,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}

	if err := tx.WithContext(ctx).Create(&newFile).Error; err != nil {
		return 0, fmt.Errorf("创建文件记录失败: %w", err)
	}
	return int64(newFile.ID), nil
}

func getFileExtension(filename string) string {
	idx := strings.LastIndex(filename, ".")
	if idx == -1 {
		return ""
	}
	return filename[idx:]
}

// Ensure DatasetItemFileRepository implements IDatasetItemFileRepository
var _ IDatasetItemFileRepository = (*DatasetItemFileRepository)(nil)

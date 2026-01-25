package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/global"
	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

const (
	// DATASET_ITEM_TTL 数据项缓存过期时间（30分钟）
	DATASET_ITEM_TTL = 30 * time.Minute
	// DATASET_ITEMS_LIST_TTL 数据项列表缓存过期时间（10分钟）
	DATASET_ITEMS_LIST_TTL = 10 * time.Minute
)

// DatasetItemService 数据集项服务
type DatasetItemService struct {
	itemRepo    repository.IDatasetItemRepository
	datasetRepo repository.IDatasetRepository
}

// NewDatasetItemService 创建数据集项服务实例
func NewDatasetItemService(itemRepo repository.IDatasetItemRepository) *DatasetItemService {
	return &DatasetItemService{itemRepo: itemRepo}
}

// NewDatasetItemServiceWithDatasetRepo 创建数据集项服务实例（包含 datasetRepo）
func NewDatasetItemServiceWithDatasetRepo(
	itemRepo repository.IDatasetItemRepository,
	datasetRepo repository.IDatasetRepository,
) *DatasetItemService {
	return &DatasetItemService{
		itemRepo:    itemRepo,
		datasetRepo: datasetRepo,
	}
}

// getDatasetRepo 获取 DatasetRepository
func (s *DatasetItemService) getDatasetRepo() repository.IDatasetRepository {
	if s.datasetRepo != nil {
		return s.datasetRepo
	}
	// 默认使用 global.DB（兼容旧代码）
	// TODO: 应该返回一个默认实现，暂时返回 nil
	return nil
}

// getRepo 获取 Repository（兼容零值实例）
func (s *DatasetItemService) getRepo() repository.IDatasetItemRepository {
	if s.itemRepo != nil {
		return s.itemRepo
	}
	return repository.NewDatasetItemRepository(global.DB)
}

// CreateDatasetItem 创建数据集项
func (datasetItemService *DatasetItemService) CreateDatasetItem(datasetId int64) (sysDatasetItem model.SysDatasetItem, err error) {
	repo := datasetItemService.getRepo()
	ctx := context.Background()

	// 校验数据集是否存在（使用注入的 datasetRepo）
	if datasetRepo := datasetItemService.getDatasetRepo(); datasetRepo != nil {
		_, err = datasetRepo.FindByID(ctx, datasetId)
		if err != nil {
			return sysDatasetItem, fmt.Errorf("数据集不存在")
		}
	} else {
		// 兼容旧代码，使用 global.DB
		var dataset model.SysDataset
		err = global.DB.Where("id = ? AND deleted = ?", datasetId, 0).First(&dataset).Error
		if err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return sysDatasetItem, fmt.Errorf("数据集不存在")
			}
			return sysDatasetItem, fmt.Errorf("查询数据集失败: %w", err)
		}
	}

	sysDatasetItem = model.SysDatasetItem{
		DatasetID: datasetId,
		Name:      "",
	}

	err = repo.Create(ctx, &sysDatasetItem)
	if err != nil {
		return sysDatasetItem, fmt.Errorf("创建数据项失败: %w", err)
	}

	// 失效统计缓存
	datasetItemService.invalidateDatasetStatsCache(datasetId)

	return sysDatasetItem, nil
}

// CreateDatasetItemWithName 创建带名称的数据集项
func (datasetItemService *DatasetItemService) CreateDatasetItemWithName(datasetId int64, itemName string) (sysDatasetItem model.SysDatasetItem, err error) {
	repo := datasetItemService.getRepo()
	ctx := context.Background()

	// 校验数据集是否存在（使用注入的 datasetRepo）
	if datasetRepo := datasetItemService.getDatasetRepo(); datasetRepo != nil {
		_, err = datasetRepo.FindByID(ctx, datasetId)
		if err != nil {
			return sysDatasetItem, fmt.Errorf("数据集不存在")
		}
	} else {
		// 兼容旧代码，使用 global.DB
		var dataset model.SysDataset
		err = global.DB.Where("id = ? AND deleted = ?", datasetId, 0).First(&dataset).Error
		if err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return sysDatasetItem, fmt.Errorf("数据集不存在")
			}
			return sysDatasetItem, fmt.Errorf("查询数据集失败: %w", err)
		}
	}

	sysDatasetItem = model.SysDatasetItem{
		DatasetID: datasetId,
		Name:      itemName,
	}

	err = repo.Create(ctx, &sysDatasetItem)
	if err != nil {
		return sysDatasetItem, fmt.Errorf("创建数据项失败: %w", err)
	}

	// 失效统计缓存
	datasetItemService.invalidateDatasetStatsCache(datasetId)

	return sysDatasetItem, nil
}

// GetDatasetItemsByDatasetID 获取数据集下的所有数据项
// 支持缓存，TTL 10分钟
func (datasetItemService *DatasetItemService) GetDatasetItemsByDatasetID(datasetID int64) ([]model.SysDatasetItem, error) {
	repo := datasetItemService.getRepo()
	ctx := context.Background()
	cacheKey := fmt.Sprintf("dataset:items:%d", datasetID)

	if global.REDIS != nil {
		// 1. 尝试从缓存获取
		cachedData, err := global.REDIS.Get(ctx, cacheKey).Result()
		if err == nil && cachedData != "" {
			var items []model.SysDatasetItem
			if err := json.Unmarshal([]byte(cachedData), &items); err == nil {
				logger.Debug("数据项列表命中缓存", zap.Int64("datasetID", datasetID))
				return items, nil
			}
		}
	}

	// 2. 从数据库查询
	items, err := repo.FindByDatasetID(ctx, datasetID)
	if err != nil {
		return nil, fmt.Errorf("查询数据项失败: %w", err)
	}

	// 3. 写入缓存
	if global.REDIS != nil {
		if itemsJSON, marshalErr := json.Marshal(items); marshalErr == nil {
			global.REDIS.Set(ctx, cacheKey, itemsJSON, DATASET_ITEMS_LIST_TTL)
		}
	}

	return items, nil
}

// GetDatasetItemsByPage 分页查询数据项列表
func (datasetItemService *DatasetItemService) GetDatasetItemsByPage(pageNum, pageSize int, datasetId int64, sceneType string) ([]*vo.ImageItemVO, int64, error) {
	repo := datasetItemService.getRepo()
	ctx := context.Background()

	if sceneType != "" {
		// 目前不按场景类型筛选，因为数据项模型中没有该字段
	}

	items, total, err := repo.FindPage(ctx, datasetId, pageNum, pageSize)
	if err != nil {
		return nil, 0, fmt.Errorf("查询数据项失败: %w", err)
	}

	// 转换为 VO
	result := make([]*vo.ImageItemVO, 0, len(items))
	itemFileService := ItemFileService{}
	for _, item := range items {
		// 获取关联文件
		hazyImages, err := itemFileService.GetImageUrlVOs(item.ID)
		if err != nil {
			logger.Warn("获取数据项文件失败", zap.Int64("itemID", item.ID), zap.Error(err))
			hazyImages = []vo.ImageUrlVO{}
		}

		itemVO := &vo.ImageItemVO{
			ID:         item.ID,
			DatasetID:  item.DatasetID,
			Name:       item.Name,
			ImageCount: len(hazyImages),
			HazyImages: hazyImages,
			CreateTime: item.CreatedAt.Format("2006-01-02 15:04:05"),
			UpdateTime: item.UpdatedAt.Format("2006-01-02 15:04:05"),
		}
		result = append(result, itemVO)
	}

	return result, total, nil
}

// GetDatasetItemById 根据ID获取数据集项（带缓存）
func (datasetItemService *DatasetItemService) GetDatasetItemById(datasetItemId int64) (sysDatasetItem model.SysDatasetItem, err error) {
	repo := datasetItemService.getRepo()
	ctx := context.Background()
	cacheKey := fmt.Sprintf("dataset:item:%d", datasetItemId)

	if global.REDIS != nil {
		// 1. 尝试从缓存获取
		cachedData, err := global.REDIS.Get(ctx, cacheKey).Result()
		if err == nil && cachedData != "" {
			if err := json.Unmarshal([]byte(cachedData), &sysDatasetItem); err == nil {
				logger.Debug("数据项命中缓存", zap.Int64("itemID", datasetItemId))
				return sysDatasetItem, nil
			}
		}
	}

	// 2. 从数据库查询
	item, err := repo.FindByID(ctx, datasetItemId)
	if err != nil {
		return sysDatasetItem, fmt.Errorf("查询数据项失败: %w", err)
	}
	if item == nil {
		return sysDatasetItem, fmt.Errorf("数据项不存在")
	}
	sysDatasetItem = *item

	// 3. 写入缓存
	if global.REDIS != nil {
		if itemJSON, marshalErr := json.Marshal(sysDatasetItem); marshalErr == nil {
			global.REDIS.Set(ctx, cacheKey, itemJSON, DATASET_ITEM_TTL)
		}
	}

	return sysDatasetItem, nil
}

// GetDatasetItemVOByID 根据ID获取数据项VO（带文件信息）
func (datasetItemService *DatasetItemService) GetDatasetItemVOByID(itemID int64) (*vo.ImageItemVO, error) {
	// 查询数据项
	item, err := datasetItemService.GetDatasetItemById(itemID)
	if err != nil {
		return nil, err
	}

	// 获取关联文件
	itemFileService := ItemFileService{}
	hazyImages, err := itemFileService.GetImageUrlVOs(itemID)
	if err != nil {
		logger.Warn("获取数据项文件失败", zap.Int64("itemID", itemID), zap.Error(err))
		hazyImages = []vo.ImageUrlVO{}
	}

	vo := &vo.ImageItemVO{
		ID:         item.ID,
		DatasetID:  item.DatasetID,
		Name:       item.Name,
		ImageCount: len(hazyImages),
		HazyImages: hazyImages,
		CreateTime: item.CreatedAt.Format("2006-01-02 15:04:05"),
		UpdateTime: item.UpdatedAt.Format("2006-01-02 15:04:05"),
	}

	return vo, nil
}

// DeleteDatasetItem 删除数据集项
func (datasetItemService *DatasetItemService) DeleteDatasetItem(datasetItemId int64) (err error) {
	repo := datasetItemService.getRepo()
	ctx := context.Background()

	// 先查询数据项
	item, err := repo.FindByID(ctx, datasetItemId)
	if err != nil || item == nil {
		return fmt.Errorf("数据项不存在")
	}

	// 删除关联的项文件（这里暂时保持原样，后续需要改造 ItemFileService）
	// TODO: ItemFileService 也需要改造为依赖注入模式
	if global.DB != nil {
		itemFileService := ItemFileService{}
		err = itemFileService.DeleteItemFileByItemId(datasetItemId)
		if err != nil {
			return fmt.Errorf("删除项文件失败: %w", err)
		}
	}

	// 删除数据项本身
	err = repo.Delete(ctx, []int64{datasetItemId})
	if err != nil {
		return fmt.Errorf("删除数据项失败: %w", err)
	}

	// 失效缓存
	datasetItemService.invalidateItemCache(item.ID)
	datasetItemService.invalidateDatasetStatsCache(item.DatasetID)

	return nil
}

// UpdateDatasetItem 更新数据集项
func (datasetItemService *DatasetItemService) UpdateDatasetItem(datasetItemId int64, itemName string) (err error) {
	repo := datasetItemService.getRepo()
	ctx := context.Background()

	// 先查询数据项
	item, err := repo.FindByID(ctx, datasetItemId)
	if err != nil || item == nil {
		return fmt.Errorf("数据项不存在")
	}

	item.Name = itemName
	err = repo.Update(ctx, item)
	if err != nil {
		return fmt.Errorf("更新数据项失败: %w", err)
	}

	// 失效缓存
	datasetItemService.invalidateItemCache(item.ID)

	return nil
}

// BatchUpdateDatasetItems 批量更新数据项
func (datasetItemService *DatasetItemService) BatchUpdateDatasetItems(updates map[int64]string) (int, error) {
	if len(updates) == 0 {
		return 0, nil
	}

	var updatedCount int
	itemIDs := make([]int64, 0, len(updates))

	for itemID, itemName := range updates {
		err := datasetItemService.UpdateDatasetItem(itemID, itemName)
		if err != nil {
			logger.Warn("更新数据项失败", zap.Int64("itemID", itemID), zap.Error(err))
			continue
		}
		updatedCount++
		itemIDs = append(itemIDs, itemID)
	}

	// 批量失效缓存
	for _, itemID := range itemIDs {
		datasetItemService.invalidateItemCache(itemID)
	}

	return updatedCount, nil
}

// ========== 缓存相关 ==========

// invalidateItemCache 失效数据项缓存
func (datasetItemService *DatasetItemService) invalidateItemCache(itemID int64) {
	if global.REDIS == nil {
		return
	}
	ctx := context.Background()
	cacheKey := fmt.Sprintf("dataset:item:%d", itemID)
	if err := global.REDIS.Del(ctx, cacheKey).Err(); err != nil {
		logger.Warn("失效数据项缓存失败", zap.String("key", cacheKey), zap.Error(err))
	}
}

// invalidateDatasetItemsCache 失效数据集下所有数据项缓存
func (datasetItemService *DatasetItemService) invalidateDatasetItemsCache(datasetID int64) {
	if global.REDIS == nil {
		return
	}
	ctx := context.Background()
	cacheKey := fmt.Sprintf("dataset:items:%d", datasetID)
	if err := global.REDIS.Del(ctx, cacheKey).Err(); err != nil {
		logger.Warn("失效数据项列表缓存失败", zap.String("key", cacheKey), zap.Error(err))
	}
}

// invalidateDatasetStatsCache 失效数据集统计缓存
func (datasetItemService *DatasetItemService) invalidateDatasetStatsCache(datasetID int64) {
	if global.REDIS == nil {
		return
	}
	ctx := context.Background()
	cacheKey := "dataset:stats:" + fmt.Sprintf("%d", datasetID)

	// 同时失效统计和叶子节点缓存
	global.REDIS.Del(ctx, cacheKey)
	leafCacheKey := "dataset:leaf:" + fmt.Sprintf("%d", datasetID)
	global.REDIS.Del(ctx, leafCacheKey)
}

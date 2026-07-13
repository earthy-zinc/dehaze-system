package dataset

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	datasetrepo "github.com/earthyzinc/dehaze-go/internal/repository/dataset"
	filerepo "github.com/earthyzinc/dehaze-go/internal/repository/file"
	fileservice "github.com/earthyzinc/dehaze-go/internal/service/file"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"go.uber.org/zap"
)

const (
	// DATASET_ITEM_TTL 数据项缓存过期时间（30分钟）
	DATASET_ITEM_TTL = 30 * time.Minute
	// DATASET_ITEMS_LIST_TTL 数据项列表缓存过期时间（10分钟）
	DATASET_ITEMS_LIST_TTL = 10 * time.Minute
)

// DatasetItemService 数据集项服务
type DatasetItemService struct {
	cache types.ICache

	itemRepo     datasetrepo.IDatasetItemRepository
	datasetRepo  datasetrepo.IDatasetRepository
	itemFileRepo filerepo.IItemFileRepository
	fileRepo     filerepo.IFileRepository

	itemFileService *fileservice.ItemFileService
}

// NewDatasetItemService 创建数据集项服务实例
func NewDatasetItemService(
	cache types.ICache,
	itemRepo datasetrepo.IDatasetItemRepository,
	datasetRepo datasetrepo.IDatasetRepository,
	itemFileRepo filerepo.IItemFileRepository,
	fileRepo filerepo.IFileRepository,
	itemFileService *fileservice.ItemFileService,
) *DatasetItemService {
	return &DatasetItemService{
		cache:           cache,
		itemRepo:        itemRepo,
		datasetRepo:     datasetRepo,
		itemFileRepo:    itemFileRepo,
		fileRepo:        fileRepo,
		itemFileService: itemFileService,
	}
}

// CreateDatasetItem 创建数据集项（可选名称）
func (datasetItemService *DatasetItemService) CreateDatasetItem(datasetId int64) (model.SysDatasetItem, error) {
	return datasetItemService.CreateDatasetItemWithName(datasetId, "")
}

// CreateDatasetItemWithName 创建带名称的数据集项
func (datasetItemService *DatasetItemService) CreateDatasetItemWithName(datasetId int64, itemName string) (sysDatasetItem model.SysDatasetItem, err error) {
	ctx := context.Background()

	// 校验数据集是否存在
	dataset, err := datasetItemService.datasetRepo.FindByID(ctx, datasetId)
	if err != nil {
		return sysDatasetItem, common.WrapBizError(common.DATABASE_ERROR, "查询数据集失败", err)
	}
	if dataset == nil {
		return sysDatasetItem, common.NewBizError(common.RESOURCE_NOT_FOUND, "数据集不存在")
	}

	sysDatasetItem = model.SysDatasetItem{
		DatasetID: datasetId,
		Name:      itemName,
	}

	err = datasetItemService.itemRepo.Create(ctx, &sysDatasetItem)
	if err != nil {
		return sysDatasetItem, common.WrapBizError(common.DATABASE_ERROR, "创建数据项失败", err)
	}

	// 失效统计和列表缓存
	datasetItemService.invalidateDatasetStatsCache(datasetId)
	datasetItemService.invalidateDatasetItemsCache(datasetId)

	return sysDatasetItem, nil
}

// GetDatasetItemsByDatasetID 获取数据集下的所有数据项
// 支持缓存，TTL 10分钟
func (datasetItemService *DatasetItemService) GetDatasetItemsByDatasetID(datasetID int64) ([]model.SysDatasetItem, error) {
	ctx := context.Background()
	cacheKey := fmt.Sprintf("dataset:items:%d", datasetID)

	if datasetItemService.cache != nil {
		// 1. 尝试从缓存获取
		cachedData, err := datasetItemService.cache.Get(ctx, cacheKey)
		if err == nil && cachedData != "" {
			var items []model.SysDatasetItem
			if err := json.Unmarshal([]byte(cachedData), &items); err == nil {
				logger.Debug("数据项列表命中缓存", zap.Int64("datasetID", datasetID))
				return items, nil
			}
		}
	}

	// 2. 从数据库查询
	items, err := datasetItemService.itemRepo.FindByDatasetID(ctx, datasetID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询数据项失败", err)
	}

	// 3. 写入缓存
	if datasetItemService.cache != nil {
		if itemsJSON, marshalErr := json.Marshal(items); marshalErr == nil {
			_ = datasetItemService.cache.Set(ctx, cacheKey, itemsJSON, DATASET_ITEMS_LIST_TTL)
		}
	}

	return items, nil
}

// GetDatasetItemsByPage 分页查询数据项列表（支持关键字搜索、雾霾程度和场景类型筛选）
// 使用批量查询避免 N+1 问题：一次性获取所有数据项的关联文件
func (datasetItemService *DatasetItemService) GetDatasetItemsByPage(pageNum, pageSize int, datasetId int64, sceneType, keyword, hazeLevel string) ([]*vo.ImageItemVO, int64, error) {
	ctx := context.Background()

	items, total, err := datasetItemService.itemRepo.FindPage(ctx, datasetId, pageNum, pageSize)
	if err != nil {
		return nil, 0, common.WrapBizError(common.DATABASE_ERROR, "查询数据项分页列表失败", err)
	}

	// 应用关键字搜索过滤（内存过滤，小数据量场景可用）
	if keyword != "" {
		items = filterItemsByKeyword(items, keyword)
		total = int64(len(items))
	}
	if hazeLevel != "" {
		items = filterItemsByHazeLevel(items, hazeLevel, datasetItemService.itemFileRepo)
		total = int64(len(items))
	}
	if sceneType != "" {
		items = filterItemsBySceneType(items, sceneType, datasetItemService.itemFileRepo)
		total = int64(len(items))
	}

	if len(items) == 0 {
		return []*vo.ImageItemVO{}, total, nil
	}

	// 收集所有数据项 ID，批量查询关联文件（避免 N+1）
	itemIDs := make([]int64, 0, len(items))
	for _, item := range items {
		itemIDs = append(itemIDs, item.ID)
	}

	allItemFiles, err := datasetItemService.itemFileRepo.FindByItemIDs(ctx, itemIDs)
	if err != nil {
		logger.Warn("批量查询数据项文件失败", zap.Error(err))
		allItemFiles = nil
	}

	// 按 item_id 分组
	itemFilesMap := make(map[int64][]model.SysItemFile)
	fileIDSet := make(map[int64]struct{})
	for _, itemFile := range allItemFiles {
		itemFilesMap[itemFile.ItemID] = append(itemFilesMap[itemFile.ItemID], itemFile)
		fileIDSet[itemFile.FileID] = struct{}{}
	}

	// 批量查询文件信息（URL 等）
	fileURLMap := make(map[int64]string)
	fileInfoMap := make(map[int64]*model.SysFile)
	if len(fileIDSet) > 0 {
		fileIDs := make([]int64, 0, len(fileIDSet))
		for id := range fileIDSet {
			fileIDs = append(fileIDs, id)
		}
		files, err := datasetItemService.fileRepo.FindByIDs(ctx, fileIDs)
		if err != nil {
			logger.Warn("批量查询文件信息失败", zap.Error(err))
		} else {
			for i := range files {
				fileURLMap[int64(files[i].ID)] = utils.StringVal(files[i].URL)
				fileInfoMap[int64(files[i].ID)] = &files[i]
			}
		}
	}

	// 组装 VO
	result := make([]*vo.ImageItemVO, 0, len(items))
	for _, item := range items {
		itemFiles := itemFilesMap[item.ID]
		imageUrls := make([]vo.ImageUrlVO, 0, len(itemFiles))

		var sceneTypeStr, descriptionStr string
		var clearImage *vo.ImageUrlVO
		for _, itemFile := range itemFiles {
			url := fileURLMap[itemFile.FileID]
			fileInfo := fileInfoMap[itemFile.FileID]

			imageUrlVO := vo.ImageUrlVO{
				ID:          itemFile.ID,
				ItemID:      itemFile.ItemID,
				DatasetID:   item.DatasetID,
				Type:        itemFile.Type,
				URL:         url,
				OriginURL:   url,
				Description: utils.StringVal(itemFile.Description),
				SceneType:   utils.StringVal(itemFile.SceneType),
				HazeLevel:   utils.StringVal(itemFile.HazeLevel),
			}

			if itemFile.Width != nil {
				imageUrlVO.Width = *itemFile.Width
			}
			if itemFile.Height != nil {
				imageUrlVO.Height = *itemFile.Height
			}

			if fileInfo != nil {
				imageUrlVO.FileName = fileInfo.Name
				if size, parseErr := strconv.ParseInt(fileInfo.Size, 10, 64); parseErr == nil {
					imageUrlVO.SizeBytes = size
				}
				// 从文件名提取格式
				if idx := strings.LastIndex(fileInfo.Name, "."); idx != -1 {
					imageUrlVO.Format = fileInfo.Name[idx+1:]
				}
			}

			if itemFile.Type == "clear" {
				clearImage = &imageUrlVO
			} else {
				imageUrls = append(imageUrls, imageUrlVO)
			}

			// 从第一个有 sceneType 的文件提取场景类型
			if sceneTypeStr == "" && itemFile.SceneType != nil {
				sceneTypeStr = utils.StringVal(itemFile.SceneType)
			}
			if descriptionStr == "" && itemFile.Description != nil {
				descriptionStr = utils.StringVal(itemFile.Description)
			}
		}

		itemVO := &vo.ImageItemVO{
			ID:          item.ID,
			DatasetID:   item.DatasetID,
			Name:        item.Name,
			SceneType:   sceneTypeStr,
			Description: descriptionStr,
			ImageCount:  len(imageUrls) + boolToInt(clearImage != nil),
			ClearImage:  clearImage,
			HazyImages:  imageUrls,
			CreateTime:  item.CreatedAt.Format("2006-01-02 15:04:05"),
			UpdateTime:  item.UpdatedAt.Format("2006-01-02 15:04:05"),
		}
		result = append(result, itemVO)
	}

	return result, total, nil
}

// boolToInt 将 bool 转换为 int
func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// GetDatasetItemById 根据ID获取数据集项（带缓存）
func (datasetItemService *DatasetItemService) GetDatasetItemById(datasetItemId int64) (sysDatasetItem model.SysDatasetItem, err error) {
	ctx := context.Background()
	cacheKey := fmt.Sprintf("dataset:item:%d", datasetItemId)

	if datasetItemService.cache != nil {
		// 1. 尝试从缓存获取
		cachedData, err := datasetItemService.cache.Get(ctx, cacheKey)
		if err == nil && cachedData != "" {
			if err := json.Unmarshal([]byte(cachedData), &sysDatasetItem); err == nil {
				logger.Debug("数据项命中缓存", zap.Int64("itemID", datasetItemId))
				return sysDatasetItem, nil
			}
		}
	}

	// 2. 从数据库查询
	item, err := datasetItemService.itemRepo.FindByID(ctx, datasetItemId)
	if err != nil {
		return sysDatasetItem, common.WrapBizError(common.DATABASE_ERROR, "查询数据项失败", err)
	}
	if item == nil {
		return sysDatasetItem, common.NewBizError(common.RESOURCE_NOT_FOUND, "数据项不存在")
	}
	sysDatasetItem = *item

	// 3. 写入缓存
	if datasetItemService.cache != nil {
		if itemJSON, marshalErr := json.Marshal(sysDatasetItem); marshalErr == nil {
			_ = datasetItemService.cache.Set(ctx, cacheKey, itemJSON, DATASET_ITEM_TTL)
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
	itemFileService := datasetItemService.itemFileService
	allImages, err := itemFileService.GetImageUrlVOs(itemID)
	if err != nil {
		logger.Warn("获取数据项文件失败", zap.Int64("itemID", itemID), zap.Error(err))
		allImages = []vo.ImageUrlVO{}
	}

	// 分离清晰图和有雾图
	var clearImage *vo.ImageUrlVO
	hazyImages := make([]vo.ImageUrlVO, 0, len(allImages))
	var sceneTypeStr, descriptionStr string

	for idx := range allImages {
		img := allImages[idx]
		if img.Type == "clear" {
			clearImage = &img
		} else {
			hazyImages = append(hazyImages, img)
		}
		if sceneTypeStr == "" && img.SceneType != "" {
			sceneTypeStr = img.SceneType
		}
		if descriptionStr == "" && img.Description != "" {
			descriptionStr = img.Description
		}
	}

	imageCount := len(hazyImages)
	if clearImage != nil {
		imageCount++
	}

	itemVO := &vo.ImageItemVO{
		ID:          item.ID,
		DatasetID:   item.DatasetID,
		Name:        item.Name,
		SceneType:   sceneTypeStr,
		Description: descriptionStr,
		ImageCount:  imageCount,
		ClearImage:  clearImage,
		HazyImages:  hazyImages,
		CreateTime:  item.CreatedAt.Format("2006-01-02 15:04:05"),
		UpdateTime:  item.UpdatedAt.Format("2006-01-02 15:04:05"),
	}

	return itemVO, nil
}

// DeleteDatasetItem 删除数据集项
func (datasetItemService *DatasetItemService) DeleteDatasetItem(datasetItemId int64) (err error) {
	ctx := context.Background()

	// 先查询数据项
	item, err := datasetItemService.itemRepo.FindByID(ctx, datasetItemId)
	if err != nil || item == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "数据项不存在")
	}

	// 删除关联的项文件（由 ItemFileService 负责）
	itemFileService := datasetItemService.itemFileService
	err = itemFileService.DeleteItemFileByItemId(datasetItemId)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除项文件失败", err)
	}

	// 删除数据项本身
	err = datasetItemService.itemRepo.Delete(ctx, []int64{datasetItemId})
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除数据项失败", err)
	}

	// 失效缓存
	datasetItemService.invalidateItemCache(item.ID)
	datasetItemService.invalidateDatasetItemsCache(item.DatasetID)
	datasetItemService.invalidateDatasetStatsCache(item.DatasetID)

	return nil
}

// UpdateDatasetItem 更新数据集项，返回更新后的 VO
func (datasetItemService *DatasetItemService) UpdateDatasetItem(datasetItemId int64, itemName string) (*vo.ImageItemVO, error) {
	ctx := context.Background()

	// 先查询数据项
	item, err := datasetItemService.itemRepo.FindByID(ctx, datasetItemId)
	if err != nil || item == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "数据项不存在")
	}

	item.Name = itemName
	err = datasetItemService.itemRepo.Update(ctx, item)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "更新数据项失败", err)
	}

	// 失效缓存
	datasetItemService.invalidateItemCache(item.ID)

	// 返回更新后的 VO
	return datasetItemService.GetDatasetItemVOByID(datasetItemId)
}

// BatchUpdateDatasetItems 批量更新数据项
func (datasetItemService *DatasetItemService) BatchUpdateDatasetItems(updates map[int64]string) (int, error) {
	if len(updates) == 0 {
		return 0, nil
	}

	var updatedCount int
	itemIDs := make([]int64, 0, len(updates))

	for itemID, itemName := range updates {
		_, err := datasetItemService.UpdateDatasetItem(itemID, itemName)
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
	if datasetItemService.cache == nil {
		return
	}
	ctx := context.Background()
	cacheKey := fmt.Sprintf("dataset:item:%d", itemID)
	if err := datasetItemService.cache.Delete(ctx, cacheKey); err != nil {
		logger.Warn("失效数据项缓存失败", zap.String("key", cacheKey), zap.Error(err))
	}
}

// invalidateDatasetItemsCache 失效数据集下所有数据项缓存
func (datasetItemService *DatasetItemService) invalidateDatasetItemsCache(datasetID int64) {
	if datasetItemService.cache == nil {
		return
	}
	ctx := context.Background()
	cacheKey := fmt.Sprintf("dataset:items:%d", datasetID)
	if err := datasetItemService.cache.Delete(ctx, cacheKey); err != nil {
		logger.Warn("失效数据项列表缓存失败", zap.String("key", cacheKey), zap.Error(err))
	}
}

// invalidateDatasetStatsCache 失效数据集统计缓存
func (datasetItemService *DatasetItemService) invalidateDatasetStatsCache(datasetID int64) {
	if datasetItemService.cache == nil {
		return
	}
	ctx := context.Background()
	keys := []string{
		fmt.Sprintf("dataset:stats:%d", datasetID),
		fmt.Sprintf("dataset:leaf:%d", datasetID),
		"dataset:statsMap:all",
		"dataset:all",
		"dataset:tree",
		"dataset:tree:options",
	}
	for _, key := range keys {
		if err := datasetItemService.cache.Delete(ctx, key); err != nil {
			logger.Warn("失效缓存失败", zap.String("key", key), zap.Error(err))
		}
	}
}

// filterItemsByKeyword 按关键字过滤数据项（文件名匹配）
func filterItemsByKeyword(items []model.SysDatasetItem, keyword string) []model.SysDatasetItem {
	result := make([]model.SysDatasetItem, 0)
	lowerKeyword := strings.ToLower(keyword)
	for _, item := range items {
		if strings.Contains(strings.ToLower(item.Name), lowerKeyword) {
			result = append(result, item)
		}
	}
	return result
}

// filterItemsByHazeLevel 按雾霾程度过滤（需关联 item_file 表）
func filterItemsByHazeLevel(items []model.SysDatasetItem, hazeLevel string, itemFileRepo filerepo.IItemFileRepository) []model.SysDatasetItem {
	if len(items) == 0 {
		return items
	}
	ctx := context.Background()
	ids := make([]int64, len(items))
	for i, item := range items {
		ids[i] = item.ID
	}
	itemFiles, err := itemFileRepo.FindByItemIDs(ctx, ids)
	if err != nil {
		logger.Warn("查询雾霾程度失败", zap.Error(err))
		return items
	}
	matchedIDs := make(map[int64]bool)
	for _, itemFile := range itemFiles {
		if itemFile.HazeLevel != nil && strings.EqualFold(*itemFile.HazeLevel, hazeLevel) {
			matchedIDs[itemFile.ItemID] = true
		}
	}
	result := make([]model.SysDatasetItem, 0)
	for _, item := range items {
		if matchedIDs[item.ID] {
			result = append(result, item)
		}
	}
	return result
}

// filterItemsBySceneType 按场景类型过滤（需关联 item_file 表）
func filterItemsBySceneType(items []model.SysDatasetItem, sceneType string, itemFileRepo filerepo.IItemFileRepository) []model.SysDatasetItem {
	if len(items) == 0 {
		return items
	}
	ctx := context.Background()
	ids := make([]int64, len(items))
	for i, item := range items {
		ids[i] = item.ID
	}
	itemFiles, err := itemFileRepo.FindByItemIDs(ctx, ids)
	if err != nil {
		logger.Warn("查询场景类型失败", zap.Error(err))
		return items
	}
	matchedIDs := make(map[int64]bool)
	for _, itemFile := range itemFiles {
		if itemFile.SceneType != nil && strings.EqualFold(*itemFile.SceneType, sceneType) {
			matchedIDs[itemFile.ItemID] = true
		}
	}
	result := make([]model.SysDatasetItem, 0)
	for _, item := range items {
		if matchedIDs[item.ID] {
			result = append(result, item)
		}
	}
	return result
}

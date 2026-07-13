package file

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	datasetrepo "github.com/earthyzinc/dehaze-go/internal/repository/dataset"
	filerepo "github.com/earthyzinc/dehaze-go/internal/repository/file"
	taskservice "github.com/earthyzinc/dehaze-go/internal/service/task"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"go.uber.org/zap"
)

const (
	// ITEM_FILE_TTL 项文件缓存过期时间（30分钟）
	ITEM_FILE_TTL = 30 * time.Minute
)

// ItemFileService 项文件服务
type ItemFileService struct {
	cache types.ICache

	itemFileRepo   filerepo.IItemFileRepository
	datasetItemRepo datasetrepo.IDatasetItemRepository
	fileService    *FileService

	taskExecutor taskservice.AsyncTaskExecutor
}

// NewItemFileService 创建项文件服务实例
func NewItemFileService(
	cache types.ICache,
	itemFileRepo filerepo.IItemFileRepository,
	datasetItemRepo datasetrepo.IDatasetItemRepository,
	fileService *FileService,
	taskExecutor taskservice.AsyncTaskExecutor,
) *ItemFileService {
	return &ItemFileService{
		cache:           cache,
		itemFileRepo:    itemFileRepo,
		datasetItemRepo: datasetItemRepo,
		fileService:     fileService,
		taskExecutor:    taskExecutor,
	}
}

// SaveItemFile 保存项文件
// sysFile: 已上传完成的文件记录（由 API 层调用 FileService.UploadFile 获取）
func (itemFileService *ItemFileService) SaveItemFile(itemId int64, sysFile model.SysFile, itemBO bo.DatasetItemBO, asyncThumbnail bool) (imageUrlVO vo.ImageUrlVO, err error) {
	ctx := context.Background()

	// 创建项文件关联记录
	sysItemFile := model.SysItemFile{
		ItemID:      itemId,
		FileID:      int64(sysFile.ID),
		Type:        itemBO.Type,
		Description: utils.StringPtr(itemBO.Description),
	}
	if itemBO.SceneType != "" {
		sysItemFile.SceneType = utils.StringPtr(itemBO.SceneType)
	}
	if itemBO.HazeLevel != "" {
		sysItemFile.HazeLevel = utils.StringPtr(itemBO.HazeLevel)
	}

	err = itemFileService.itemFileRepo.Create(ctx, &sysItemFile)
	if err != nil {
		return imageUrlVO, common.WrapBizError(common.DATABASE_ERROR, "创建项文件关联失败", err)
	}

	// 异步生成缩略图
	if asyncThumbnail {
		itemFileService.submitThumbnailTask(itemId, int64(sysFile.ID), sysItemFile.ID)
	}

	// 查询数据项以获取 datasetId
	var datasetID int64
	datasetItem, err := itemFileService.datasetItemRepo.FindByID(ctx, itemId)
	if err == nil && datasetItem != nil {
		datasetID = datasetItem.DatasetID
	}

	// 构建返回对象
	imageUrlVO = vo.ImageUrlVO{
		ID:          sysItemFile.ID,
		ItemID:      itemId,
		DatasetID:   datasetID,
		Type:        sysItemFile.Type,
		URL:         utils.StringVal(sysFile.URL),
		OriginURL:   utils.StringVal(sysFile.URL),
		Description: utils.StringVal(sysItemFile.Description),
		SceneType:   utils.StringVal(sysItemFile.SceneType),
		HazeLevel:   utils.StringVal(sysItemFile.HazeLevel),
		FileName:    sysFile.Name,
	}

	// 解析文件大小
	if size, parseErr := strconv.ParseInt(sysFile.Size, 10, 64); parseErr == nil {
		imageUrlVO.SizeBytes = size
	}

	// 从文件名提取格式
	if idx := strings.LastIndex(sysFile.Name, "."); idx != -1 {
		imageUrlVO.Format = sysFile.Name[idx+1:]
	}

	// 设置宽高
	if sysItemFile.Width != nil {
		imageUrlVO.Width = *sysItemFile.Width
	}
	if sysItemFile.Height != nil {
		imageUrlVO.Height = *sysItemFile.Height
	}

	// 失效缓存
	itemFileService.invalidateItemFilesCache(itemId)
	if datasetID > 0 {
		itemFileService.invalidateDatasetStatsCache(datasetID)
	}

	return imageUrlVO, nil
}

// GetImageUrlVOs 获取图片URL VO列表（带缓存）
func (itemFileService *ItemFileService) GetImageUrlVOs(itemId int64) (imageUrlVOs []vo.ImageUrlVO, err error) {
	ctx := context.Background()
	cacheKey := fmt.Sprintf("item:files:%d", itemId)

	// 1. 尝试从缓存获取
	if itemFileService.cache != nil {
		cachedData, err := itemFileService.cache.Get(ctx, cacheKey)
		if err == nil && cachedData != "" {
			if err := json.Unmarshal([]byte(cachedData), &imageUrlVOs); err == nil {
				logger.Debug("项文件列表命中缓存", zap.Int64("itemID", itemId))
				return imageUrlVOs, nil
			}
		}
	}

	// 2. 从数据库查询
	sysItemFiles, err := itemFileService.itemFileRepo.FindByItemID(ctx, itemId)
	if err != nil {
		return imageUrlVOs, common.WrapBizError(common.DATABASE_ERROR, "查询项文件失败", err)
	}

	imageUrlVOs = make([]vo.ImageUrlVO, 0, len(sysItemFiles))

	// 查询数据项以获取 datasetId
	var datasetID int64
	datasetItem, err := itemFileService.datasetItemRepo.FindByID(ctx, itemId)
	if err == nil && datasetItem != nil {
		datasetID = datasetItem.DatasetID
	}

	// 批量收集所有文件 ID（含缩略图 ID），一次性查询消除 N+1
	fileIDSet := make(map[int64]struct{})
	for _, itemFile := range sysItemFiles {
		fileIDSet[itemFile.FileID] = struct{}{}
		if itemFile.ThumbnailFileID != nil {
			fileIDSet[*itemFile.ThumbnailFileID] = struct{}{}
		}
	}
	allFileIDs := make([]int64, 0, len(fileIDSet))
	for id := range fileIDSet {
		allFileIDs = append(allFileIDs, id)
	}
	fileMap, batchErr := itemFileService.fileService.GetFilesByIdsMap(ctx, allFileIDs)
	if batchErr != nil {
		logger.Warn("批量查询文件失败", zap.Error(batchErr))
		fileMap = map[int64]model.SysFile{}
	}

	// 获取关联的文件信息
	for _, itemFile := range sysItemFiles {
		sysFile, ok := fileMap[itemFile.FileID]
		if !ok {
			logger.Warn("文件记录不存在", zap.Int64("fileID", itemFile.FileID))
			continue
		}

		imageUrlVO := vo.ImageUrlVO{
			ID:          itemFile.ID,
			ItemID:      itemFile.ItemID,
			DatasetID:   datasetID,
			Type:        itemFile.Type,
			URL:         utils.StringVal(sysFile.URL),
			OriginURL:   utils.StringVal(sysFile.URL),
			Description: utils.StringVal(itemFile.Description),
			SceneType:   utils.StringVal(itemFile.SceneType),
			HazeLevel:   utils.StringVal(itemFile.HazeLevel),
			FileName:    sysFile.Name,
		}

		// 解析文件大小
		if size, parseErr := strconv.ParseInt(sysFile.Size, 10, 64); parseErr == nil {
			imageUrlVO.SizeBytes = size
		}

		// 从文件名提取格式
		if idx := strings.LastIndex(sysFile.Name, "."); idx != -1 {
			imageUrlVO.Format = sysFile.Name[idx+1:]
		}

		// 设置宽高
		if itemFile.Width != nil {
			imageUrlVO.Width = *itemFile.Width
		}
		if itemFile.Height != nil {
			imageUrlVO.Height = *itemFile.Height
		}

		// 设置缩略图URL
		if itemFile.ThumbnailFileID != nil {
			if thumbFile, ok := fileMap[*itemFile.ThumbnailFileID]; ok {
				imageUrlVO.ThumbnailURL = utils.StringVal(thumbFile.URL)
			}
		}

		imageUrlVOs = append(imageUrlVOs, imageUrlVO)
	}

	// 3. 写入缓存
	if itemFileService.cache != nil {
		if imageUrlVOsJSON, marshalErr := json.Marshal(imageUrlVOs); marshalErr == nil {
			_ = itemFileService.cache.Set(ctx, cacheKey, imageUrlVOsJSON, ITEM_FILE_TTL)
		}
	}

	return imageUrlVOs, nil
}

// DeleteItemFile 删除项文件
func (itemFileService *ItemFileService) DeleteItemFile(itemFileId int64) (err error) {
	ctx := context.Background()

	// 先查询项文件
	itemFile, err := itemFileService.itemFileRepo.FindByID(ctx, itemFileId)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询项文件失败", err)
	}
	if itemFile == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "项文件不存在")
	}

	// 查询数据项
	var datasetID int64
	item, err := itemFileService.datasetItemRepo.FindByID(ctx, itemFile.ItemID)
	if err != nil {
		logger.Warn("查询数据项失败", zap.Int64("itemID", itemFile.ItemID), zap.Error(err))
	} else if item != nil {
		datasetID = item.DatasetID
	}

	// 删除物理文件（异步）
	go itemFileService.deletePhysicalFileAsync(itemFile.FileID, itemFile.ThumbnailFileID)

	// 删除数据库记录
	err = itemFileService.itemFileRepo.Delete(ctx, itemFileId)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除项文件失败", err)
	}

	// 失效缓存
	itemFileService.invalidateItemFilesCache(itemFile.ItemID)
	if datasetID > 0 {
		itemFileService.invalidateDatasetStatsCache(datasetID)
	}

	return nil
}

// DeleteItemFileByItemId 根据项ID删除项文件
func (itemFileService *ItemFileService) DeleteItemFileByItemId(itemId int64) (err error) {
	ctx := context.Background()

	// 先查询所有项文件
	itemFiles, err := itemFileService.itemFileRepo.FindByItemID(ctx, itemId)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询项文件失败", err)
	}

	// 收集文件ID
	fileIDs := make([]int64, 0, len(itemFiles))
	thumbFileIDs := make([]int64, 0)
	for _, itemFile := range itemFiles {
		fileIDs = append(fileIDs, itemFile.FileID)
		if itemFile.ThumbnailFileID != nil {
			thumbFileIDs = append(thumbFileIDs, *itemFile.ThumbnailFileID)
		}
	}

	// 删除物理文件（异步）
	for _, fileID := range fileIDs {
		go itemFileService.deletePhysicalFileAsync(fileID, nil)
	}

	// 删除数据库记录
	err = itemFileService.itemFileRepo.DeleteByItemID(ctx, itemId)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除项文件失败", err)
	}

	// 失效缓存
	itemFileService.invalidateItemFilesCache(itemId)

	return nil
}

// GetItemFileById 根据ID获取项文件（带缓存），返回 ImageUrlVO
func (itemFileService *ItemFileService) GetItemFileById(itemFileId int64) (imageUrlVO vo.ImageUrlVO, err error) {
	ctx := context.Background()
	cacheKey := fmt.Sprintf("item:file:%d", itemFileId)

	// 1. 尝试从缓存获取
	if itemFileService.cache != nil {
		cachedData, err := itemFileService.cache.Get(ctx, cacheKey)
		if err == nil && cachedData != "" {
			if err := json.Unmarshal([]byte(cachedData), &imageUrlVO); err == nil {
				logger.Debug("项文件命中缓存", zap.Int64("itemFileID", itemFileId))
				return imageUrlVO, nil
			}
		}
	}

	// 2. 从数据库查询
	itemFile, err := itemFileService.itemFileRepo.FindByID(ctx, itemFileId)
	if err != nil {
		return imageUrlVO, common.WrapBizError(common.DATABASE_ERROR, "查询项文件失败", err)
	}
	if itemFile == nil {
		return imageUrlVO, common.NewBizError(common.RESOURCE_NOT_FOUND, "项文件不存在")
	}

	// 查询关联文件
	sysFile, fileErr := itemFileService.fileService.GetFileById(itemFile.FileID)
	if fileErr != nil {
		logger.Warn("查询关联文件失败", zap.Int64("fileID", itemFile.FileID), zap.Error(fileErr))
	}

	// 查询数据项以获取 datasetId
	var datasetID int64
	datasetItem, err := itemFileService.datasetItemRepo.FindByID(ctx, itemFile.ItemID)
	if err == nil && datasetItem != nil {
		datasetID = datasetItem.DatasetID
	}

	// 构建 ImageUrlVO
	imageUrlVO = vo.ImageUrlVO{
		ID:          itemFile.ID,
		ItemID:      itemFile.ItemID,
		DatasetID:   datasetID,
		Type:        itemFile.Type,
		Description: utils.StringVal(itemFile.Description),
		SceneType:   utils.StringVal(itemFile.SceneType),
		HazeLevel:   utils.StringVal(itemFile.HazeLevel),
	}

	if fileErr == nil && sysFile.ID > 0 {
		imageUrlVO.URL = utils.StringVal(sysFile.URL)
		imageUrlVO.OriginURL = utils.StringVal(sysFile.URL)
		imageUrlVO.FileName = sysFile.Name
		if size, parseErr := strconv.ParseInt(sysFile.Size, 10, 64); parseErr == nil {
			imageUrlVO.SizeBytes = size
		}
		if idx := strings.LastIndex(sysFile.Name, "."); idx != -1 {
			imageUrlVO.Format = sysFile.Name[idx+1:]
		}
	}

	if itemFile.Width != nil {
		imageUrlVO.Width = *itemFile.Width
	}
	if itemFile.Height != nil {
		imageUrlVO.Height = *itemFile.Height
	}

	// 设置缩略图URL
	if itemFile.ThumbnailFileID != nil {
		if thumbFile, err := itemFileService.fileService.GetFileById(*itemFile.ThumbnailFileID); err == nil {
			imageUrlVO.ThumbnailURL = utils.StringVal(thumbFile.URL)
		}
	}

	// 3. 写入缓存
	if itemFileService.cache != nil {
		if voJSON, marshalErr := json.Marshal(imageUrlVO); marshalErr == nil {
			_ = itemFileService.cache.Set(ctx, cacheKey, voJSON, ITEM_FILE_TTL)
		}
	}

	return imageUrlVO, nil
}

// UpdateItemFileInfo 更新图片信息，返回更新后的 VO
func (itemFileService *ItemFileService) UpdateItemFileInfo(itemFileID int64, form bo.ItemFileUpdateForm) (vo.ImageUrlVO, error) {
	ctx := context.Background()

	itemFile, err := itemFileService.itemFileRepo.FindByID(ctx, itemFileID)
	if err != nil {
		return vo.ImageUrlVO{}, common.WrapBizError(common.DATABASE_ERROR, "查询项文件失败", err)
	}
	if itemFile == nil {
		return vo.ImageUrlVO{}, common.NewBizError(common.RESOURCE_NOT_FOUND, "项文件不存在")
	}

	// 更新提供的字段
	if form.Type != nil {
		itemFile.Type = *form.Type
	}
	if form.SceneType != nil {
		itemFile.SceneType = form.SceneType
	}
	if form.HazeLevel != nil {
		itemFile.HazeLevel = form.HazeLevel
	}
	if form.Description != nil {
		itemFile.Description = form.Description
	}

	err = itemFileService.itemFileRepo.Update(ctx, itemFile)
	if err != nil {
		return vo.ImageUrlVO{}, common.WrapBizError(common.DATABASE_ERROR, "更新项文件失败", err)
	}

	// 失效缓存
	itemFileService.invalidateItemFileCache(itemFileID)
	itemFileService.invalidateItemFilesCache(itemFile.ItemID)

	// 返回更新后的 VO
	return itemFileService.GetItemFileById(itemFileID)
}

// UpdateThumbnail 更新缩略图
func (itemFileService *ItemFileService) UpdateThumbnail(itemFileID, thumbnailFileID int64) error {
	ctx := context.Background()
	err := itemFileService.itemFileRepo.UpdateThumbnail(ctx, itemFileID, thumbnailFileID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新缩略图失败", err)
	}

	// 失效缓存
	itemFileService.invalidateItemFileCache(itemFileID)

	return nil
}

// ========== 异步任务相关 ==========

// submitThumbnailTask 提交缩略图生成任务
func (itemFileService *ItemFileService) submitThumbnailTask(itemID, fileID, itemFileID int64) {
	taskIDStr := fmt.Sprintf("thumb_%d_%d", fileID, itemFileID)

	if itemFileService.taskExecutor == nil {
		logger.Error("任务执行器未初始化")
		return
	}

	payload := taskservice.ThumbnailTaskPayload{
		ItemID:     itemID,
		FileID:     fileID,
		ItemFileID: itemFileID,
	}
	msg := taskservice.TaskMessage{
		TaskID:    taskIDStr,
		TaskType:  "thumbnail",
		Total:     1,
		Payload:   payload,
		CreatedAt: time.Now(),
	}
	if err := itemFileService.taskExecutor.PublishTask(context.Background(), msg); err != nil {
		logger.Error("提交缩略图任务失败", zap.String("taskID", taskIDStr), zap.Error(err))
	}
}


// deletePhysicalFileAsync 异步删除物理文件
func (itemFileService *ItemFileService) deletePhysicalFileAsync(fileID int64, thumbFileID *int64) {
	// 查询文件路径
	file, err := itemFileService.fileService.GetFileById(fileID)
	if err != nil {
		logger.Warn("查询文件失败", zap.Int64("fileID", fileID), zap.Error(err))
		return
	}

	// TODO: 调用文件存储服务删除文件
	logger.Info("删除物理文件（TODO: 实现文件存储删除）",
		zap.Int64("fileID", fileID),
		zap.String("path", file.Path))

	// 如果有缩略图，也删除
	if thumbFileID != nil {
		thumbFile, err := itemFileService.fileService.GetFileById(*thumbFileID)
		if err == nil {
			// TODO: 删除缩略图文件
			logger.Info("删除缩略图文件",
				zap.Int64("thumbFileID", *thumbFileID),
				zap.String("path", thumbFile.Path))
		}
	}
}

// ========== 缓存相关 ==========

// invalidateItemFileCache 失效项文件缓存
func (itemFileService *ItemFileService) invalidateItemFileCache(itemFileID int64) {
	if itemFileService.cache == nil {
		return
	}
	ctx := context.Background()
	cacheKey := fmt.Sprintf("item:file:%d", itemFileID)
	_ = itemFileService.cache.Delete(ctx, cacheKey)
}

// invalidateItemFilesCache 失效数据项下所有文件缓存
func (itemFileService *ItemFileService) invalidateItemFilesCache(itemID int64) {
	if itemFileService.cache == nil {
		return
	}
	ctx := context.Background()
	cacheKey := fmt.Sprintf("item:files:%d", itemID)
	_ = itemFileService.cache.Delete(ctx, cacheKey)
}

// invalidateDatasetStatsCache 失效数据集统计缓存
func (itemFileService *ItemFileService) invalidateDatasetStatsCache(datasetID int64) {
	if itemFileService.cache == nil {
		return
	}
	ctx := context.Background()
	cacheKey := "dataset:stats:" + fmt.Sprintf("%d", datasetID)
	_ = itemFileService.cache.Delete(ctx, cacheKey)
}

// ========== 辅助函数 ==========


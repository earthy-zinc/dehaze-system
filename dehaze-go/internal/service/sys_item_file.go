package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/global"
	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/dto"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"go.uber.org/zap"
)

const (
	// ITEM_FILE_TTL 项文件缓存过期时间（30分钟）
	ITEM_FILE_TTL = 30 * time.Minute
	// THUMBNAIL_WIDTH 缩略图固定宽度
	THUMBNAIL_WIDTH = 200
)

// ItemFileService 项文件服务
type ItemFileService struct {
	itemFileRepo repository.IItemFileRepository
	taskExecutor *TaskExecutor
}

// NewItemFileService 创建项文件服务实例
func NewItemFileService(itemFileRepo repository.IItemFileRepository) *ItemFileService {
	return &ItemFileService{
		itemFileRepo: itemFileRepo,
		taskExecutor: &TaskExecutor{},
	}
}

// getRepo 获取 Repository（兼容零值实例）
func (s *ItemFileService) getRepo() repository.IItemFileRepository {
	if s.itemFileRepo != nil {
		return s.itemFileRepo
	}
	return repository.NewItemFileRepository(global.DB)
}

// SaveItemFile 保存项文件
func (itemFileService *ItemFileService) SaveItemFile(itemId int64, itemBO bo.DatasetItemBO, asyncThumbnail bool) (imageFileInfo dto.ImageFileInfo, err error) {
	repo := itemFileService.getRepo()
	ctx := context.Background()

	// 创建文件记录
	sysFileService := SysFileService{}

	fileBO := bo.FileBO{
		Name:       itemBO.Name,
		ObjectName: "",
		Extension:  itemBO.Extension,
		MD5:        "",
		Path:       "",
		Size:       itemBO.Size,
		URL:        "",
	}

	sysFile, err := sysFileService.SaveFile(fileBO)
	if err != nil {
		return imageFileInfo, fmt.Errorf("保存文件失败: %w", err)
	}

	// 创建项文件关联记录
	sysItemFile := model.SysItemFile{
		ItemID:      itemId,
		FileID:      int64(sysFile.ID),
		Type:        itemBO.Type,
		Description: utils.StringPtr(itemBO.Description),
	}

	err = repo.Create(ctx, &sysItemFile)
	if err != nil {
		return imageFileInfo, fmt.Errorf("创建项文件关联失败: %w", err)
	}

	// 异步生成缩略图
	if asyncThumbnail {
		itemFileService.submitThumbnailTask(itemId, int64(sysFile.ID), sysItemFile.ID)
	}

	// 构建返回对象
	imageFileInfo = dto.ImageFileInfo{
		ID:            sysItemFile.ID,
		DatasetItemID: itemId,
		FileID:        int64(sysFile.ID),
		Type:          sysItemFile.Type,
		Description:   utils.StringVal(sysItemFile.Description),
		URL:           utils.StringVal(sysFile.URL),
	}

	return imageFileInfo, nil
}

// GetImageUrlVOs 获取图片URL VO列表（带缓存）
func (itemFileService *ItemFileService) GetImageUrlVOs(itemId int64) (imageUrlVOs []vo.ImageUrlVO, err error) {
	repo := itemFileService.getRepo()
	ctx := context.Background()
	cacheKey := fmt.Sprintf("item:files:%d", itemId)

	// 1. 尝试从缓存获取
	if global.REDIS != nil {
		cachedData, err := global.REDIS.Get(ctx, cacheKey).Result()
		if err == nil && cachedData != "" {
			if err := json.Unmarshal([]byte(cachedData), &imageUrlVOs); err == nil {
				logger.Debug("项文件列表命中缓存", zap.Int64("itemID", itemId))
				return imageUrlVOs, nil
			}
		}
	}

	// 2. 从数据库查询
	sysItemFiles, err := repo.FindByItemID(ctx, itemId)
	if err != nil {
		return imageUrlVOs, fmt.Errorf("查询项文件失败: %w", err)
	}

	imageUrlVOs = make([]vo.ImageUrlVO, 0, len(sysItemFiles))

	// 获取关联的文件信息
	for _, itemFile := range sysItemFiles {
		var sysFile model.SysFile
		err := global.DB.Where("id = ?", itemFile.FileID).First(&sysFile).Error
		if err != nil {
			logger.Warn("查询文件失败", zap.Int64("fileID", itemFile.FileID), zap.Error(err))
			continue
		}

		imageUrlVO := vo.ImageUrlVO{
			ID:          itemFile.ID,
			Type:        itemFile.Type,
			URL:         utils.StringVal(sysFile.URL),
			OriginURL:   utils.StringVal(sysFile.URL),
			Description: utils.StringVal(itemFile.Description),
		}
		imageUrlVOs = append(imageUrlVOs, imageUrlVO)
	}

	// 3. 写入缓存
	if global.REDIS != nil {
		if imageUrlVOsJSON, marshalErr := json.Marshal(imageUrlVOs); marshalErr == nil {
			global.REDIS.Set(ctx, cacheKey, imageUrlVOsJSON, ITEM_FILE_TTL)
		}
	}

	return imageUrlVOs, nil
}

// DeleteItemFile 删除项文件
func (itemFileService *ItemFileService) DeleteItemFile(itemFileId int64) (err error) {
	repo := itemFileService.getRepo()
	ctx := context.Background()

	// 先查询项文件
	itemFile, err := repo.FindByID(ctx, itemFileId)
	if err != nil {
		return fmt.Errorf("查询项文件失败: %w", err)
	}
	if itemFile == nil {
		return fmt.Errorf("项文件不存在")
	}

	// 查询数据项ID
	var item model.SysDatasetItem
	err = global.DB.Where("id = ?", itemFile.ItemID).First(&item).Error
	if err != nil {
		logger.Warn("查询数据项失败", zap.Int64("itemID", itemFile.ItemID), zap.Error(err))
	}

	// 删除物理文件（异步）
	go itemFileService.deletePhysicalFileAsync(itemFile.FileID, itemFile.ThumbnailFileID)

	// 删除数据库记录
	err = repo.Delete(ctx, itemFileId)
	if err != nil {
		return fmt.Errorf("删除项文件失败: %w", err)
	}

	// 失效缓存
	itemFileService.invalidateItemFilesCache(itemFile.ItemID)
	if item.ID > 0 {
		itemFileService.invalidateDatasetStatsCache(item.DatasetID)
	}

	return nil
}

// DeleteItemFileByItemId 根据项ID删除项文件
func (itemFileService *ItemFileService) DeleteItemFileByItemId(itemId int64) (err error) {
	repo := itemFileService.getRepo()
	ctx := context.Background()

	// 先查询所有项文件
	itemFiles, err := repo.FindByItemID(ctx, itemId)
	if err != nil {
		return fmt.Errorf("查询项文件失败: %w", err)
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
	err = repo.DeleteByItemID(ctx, itemId)
	if err != nil {
		return fmt.Errorf("删除项文件失败: %w", err)
	}

	// 失效缓存
	itemFileService.invalidateItemFilesCache(itemId)

	return nil
}

// GetItemFileById 根据ID获取项文件（带缓存）
func (itemFileService *ItemFileService) GetItemFileById(itemFileId int64) (sysItemFile model.SysItemFile, err error) {
	repo := itemFileService.getRepo()
	ctx := context.Background()
	cacheKey := fmt.Sprintf("item:file:%d", itemFileId)

	// 1. 尝试从缓存获取
	if global.REDIS != nil {
		cachedData, err := global.REDIS.Get(ctx, cacheKey).Result()
		if err == nil && cachedData != "" {
			if err := json.Unmarshal([]byte(cachedData), &sysItemFile); err == nil {
				logger.Debug("项文件命中缓存", zap.Int64("itemFileID", itemFileId))
				return sysItemFile, nil
			}
		}
	}

	// 2. 从数据库查询
	itemFile, err := repo.FindByID(ctx, itemFileId)
	if err != nil {
		return sysItemFile, fmt.Errorf("查询项文件失败: %w", err)
	}
	if itemFile == nil {
		return sysItemFile, fmt.Errorf("项文件不存在")
	}
	sysItemFile = *itemFile

	// 3. 写入缓存
	if global.REDIS != nil {
		if itemFileJSON, marshalErr := json.Marshal(sysItemFile); marshalErr == nil {
			global.REDIS.Set(ctx, cacheKey, itemFileJSON, ITEM_FILE_TTL)
		}
	}

	return sysItemFile, nil
}

// UpdateThumbnail 更新缩略图
func (itemFileService *ItemFileService) UpdateThumbnail(itemFileID, thumbnailFileID int64) error {
	repo := itemFileService.getRepo()
	ctx := context.Background()
	err := repo.UpdateThumbnail(ctx, itemFileID, thumbnailFileID)
	if err != nil {
		return fmt.Errorf("更新缩略图失败: %w", err)
	}

	// 失效缓存
	itemFileService.invalidateItemFileCache(itemFileID)

	return nil
}

// ========== 异步任务相关 ==========

// submitThumbnailTask 提交缩略图生成任务
func (itemFileService *ItemFileService) submitThumbnailTask(itemID, fileID, itemFileID int64) {
	taskIDStr := fmt.Sprintf("thumb_%d_%d", fileID, itemFileID)

	pool := itemFileService.taskExecutor.getWorkerPool("thumbnail")
	if pool == nil {
		logger.Error("缩略图Worker池不存在")
		return
	}

	pool.SubmitWithCtx(taskIDStr, "thumbnail", 1, func(ctx context.Context) error {
		// 检查是否取消
		if itemFileService.isTaskCanceled(ctx, taskIDStr) {
			return errors.New("任务被取消")
		}

		if err := itemFileService.generateThumbnail(fileID, itemFileID); err != nil {
			logger.Error("生成缩略图失败",
				zap.Int64("fileID", fileID),
				zap.Int64("itemFileID", itemFileID),
				zap.Error(err))
			// 记录失败
			itemFileService.recordThumbnailFailure(fileID)
			return err
		}
		return nil
	})
}

// generateThumbnail 生成缩略图
func (itemFileService *ItemFileService) generateThumbnail(fileID, itemFileID int64) error {
	// 1. 查询源文件
	var sourceFile model.SysFile
	err := global.DB.Where("id = ?", fileID).First(&sourceFile).Error
	if err != nil {
		return fmt.Errorf("查询源文件失败: %w", err)
	}

	// 2. TODO: 实际生成缩略图的逻辑
	// 这里应该：
	// a. 从源路径读取图片
	// b. 使用图像处理库生成缩略图（宽度固定为200px，高度等比缩放）
	// c. 保存缩略图到文件存储
	// d. 创建新的SysFile记录
	// e. 更新SysItemFile的thumbnail_file_id

	logger.Info("生成缩略图（TODO: 实现图像处理）",
		zap.Int64("fileID", fileID),
		zap.String("sourcePath", sourceFile.Path))

	return nil
}

// deletePhysicalFileAsync 异步删除物理文件
func (itemFileService *ItemFileService) deletePhysicalFileAsync(fileID int64, thumbFileID *int64) {
	// 查询文件路径
	var file model.SysFile
	err := global.DB.Where("id = ?", fileID).First(&file).Error
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
		var thumbFile model.SysFile
		if err := global.DB.Where("id = ?", *thumbFileID).First(&thumbFile).Error; err == nil {
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
	if global.REDIS == nil {
		return
	}
	ctx := context.Background()
	cacheKey := fmt.Sprintf("item:file:%d", itemFileID)
	global.REDIS.Del(ctx, cacheKey)
}

// invalidateItemFilesCache 失效数据项下所有文件缓存
func (itemFileService *ItemFileService) invalidateItemFilesCache(itemID int64) {
	if global.REDIS == nil {
		return
	}
	ctx := context.Background()
	cacheKey := fmt.Sprintf("item:files:%d", itemID)
	global.REDIS.Del(ctx, cacheKey)
}

// invalidateDatasetStatsCache 失效数据集统计缓存
func (itemFileService *ItemFileService) invalidateDatasetStatsCache(datasetID int64) {
	if global.REDIS == nil {
		return
	}
	ctx := context.Background()
	cacheKey := "dataset:stats:" + fmt.Sprintf("%d", datasetID)
	global.REDIS.Del(ctx, cacheKey)
}

// ========== 辅助函数 ==========

// isTaskCanceled 检查任务是否被取消
func (itemFileService *ItemFileService) isTaskCanceled(ctx context.Context, taskID string) bool {
	select {
	case <-ctx.Done():
		return true
	default:
		if global.REDIS == nil {
			return false
		}
		// 检查Redis中的取消标志
		cancelKey := TASK_CANCEL_PREFIX + taskID
		canceled, _ := global.REDIS.Get(ctx, cancelKey).Result()
		return canceled == "true"
	}
}

// recordThumbnailFailure 记录缩略图生成失败
func (itemFileService *ItemFileService) recordThumbnailFailure(fileID int64) {
	if global.REDIS == nil {
		return
	}
	ctx := context.Background()
	global.REDIS.HSet(ctx, THUMBNAIL_FAILED_KEY, fileID, time.Now().Unix())
}

// ========== 缓存键常量 ==========

const (
	THUMBNAIL_FAILED_KEY = "dataset:thumbnail:failed"
)

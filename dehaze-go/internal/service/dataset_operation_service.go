package service

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/domain"
	"github.com/earthyzinc/dehaze-go/internal/global"
	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

// DatasetOperationService 数据集操作服务
// 负责处理复杂的数据集操作，如批量上传、级联删除等
type DatasetOperationService struct {
	datasetRepo     repository.IDatasetRepository
	datasetItemRepo repository.IDatasetItemRepository
	itemFileRepo    repository.IItemFileRepository
	taskExecutor    *TaskExecutor
	domain          *domain.PairedImageValidator
	treeUtils       *utils.TreeDataUtils
}

// NewDatasetOperationService 创建数据集操作服务
func NewDatasetOperationService(
	datasetRepo repository.IDatasetRepository,
	datasetItemRepo repository.IDatasetItemRepository,
	itemFileRepo repository.IItemFileRepository,
) *DatasetOperationService {
	return &DatasetOperationService{
		datasetRepo:     datasetRepo,
		datasetItemRepo: datasetItemRepo,
		itemFileRepo:    itemFileRepo,
		taskExecutor:    &TaskExecutor{},
		domain:          domain.NewPairedImageValidator(),
		treeUtils:       utils.NewTreeDataUtils(),
	}
}

// getDatasetRepo 获取 DatasetRepository（兼容零值实例）
func (dos *DatasetOperationService) getDatasetRepo() repository.IDatasetRepository {
	if dos.datasetRepo != nil {
		return dos.datasetRepo
	}
	return repository.NewDatasetRepository(global.DB)
}

// getDatasetItemRepo 获取 DatasetItemRepository（兼容零值实例）
func (dos *DatasetOperationService) getDatasetItemRepo() repository.IDatasetItemRepository {
	if dos.datasetItemRepo != nil {
		return dos.datasetItemRepo
	}
	return repository.NewDatasetItemRepository(global.DB)
}

// getItemFileRepo 获取 ItemFileRepository（兼容零值实例）
func (dos *DatasetOperationService) getItemFileRepo() repository.IItemFileRepository {
	if dos.itemFileRepo != nil {
		return dos.itemFileRepo
	}
	return repository.NewItemFileRepository(global.DB)
}

// SetRepositories 设置 Repository（测试用）
func (dos *DatasetOperationService) SetRepositories(
	datasetRepo repository.IDatasetRepository,
	datasetItemRepo repository.IDatasetItemRepository,
	itemFileRepo repository.IItemFileRepository,
) {
	dos.datasetRepo = datasetRepo
	dos.datasetItemRepo = datasetItemRepo
	dos.itemFileRepo = itemFileRepo
}

// getBatchUpdateUserID 从上下文中获取当前用户ID
// 用于批量更新操作时的 update_by 字段填充
// 修复 P0 级别批量删除逻辑错误：确保批量更新时能获取到当前用户ID
func getBatchUpdateUserID(ctx context.Context) int64 {
	// 尝试从上下文中获取用户ID（如果通过 WithUserIDWithContext 传递）
	if userID := ctx.Value("userId"); userID != nil {
		if id, ok := userID.(int64); ok {
			return id
		}
		if id, ok := userID.(int); ok {
			return int64(id)
		}
	}

	// 尝试从Gin上下文中获取（如果当前请求设置了Gin上下文）
	c := common.GetCurrentGinContext()
	if c == nil {
		return 0
	}

	// 尝试从claims中获取
	if claims, exists := c.Get("claims"); exists {
		if userClaims, ok := claims.(common.UserClaims); ok {
			return userClaims.GetUserID()
		}
	}

	// 尝试直接从上下文中获取
	if userID, exists := c.Get("userId"); exists {
		if id, ok := userID.(int64); ok {
			return id
		}
		if id, ok := userID.(int); ok {
			return int64(id)
		}
	}

	return 0
}

// CreateDatasetItemWithImagesRequest 创建数据项请求
type CreateDatasetItemWithImagesRequest struct {
	DatasetID  int64             `json:"datasetId" validate:"required"`
	ItemName   string            `json:"itemName"`
	SceneType  string            `json:"sceneType"`
	ClearImage ImageUploadInfo   `json:"clearImage"`
	HazyImages []ImageUploadInfo `json:"hazyImages"`
	Options    CreateItemOptions `json:"options"`
}

// ImageUploadInfo 图片上传信息
type ImageUploadInfo struct {
	Type      string `json:"type"`
	Name      string `json:"name"`
	Path      string `json:"path"`
	URL       string `json:"url"`
	Size      int64  `json:"size"`
	MD5       string `json:"md5"`
	Width     int    `json:"width"`
	Height    int    `json:"height"`
	HazeLevel string `json:"hazeLevel,omitempty"`
}

// CreateItemOptions 创建项选项
type CreateItemOptions struct {
	ValidateResolution bool `json:"validateResolution"` // 是否校验分辨率
	SkipThumbnail      bool `json:"skipThumbnail"`      // 是否跳过缩略图生成
	AsyncThumbnail     bool `json:"asyncThumbnail"`     // 是否异步生成缩略图
}

// CreateDatasetItemWithImages 创建数据项并上传配对图片
// 这是一个复杂操作，包含：
// 1. 创建数据集项
// 2. 上传清晰图文件
// 3. 上传雾霾图文件
// 4. 配对图片分辨率校验（可选）
// 5. 生成缩略图（异步）
// 6. 更新统计缓存
func (dos *DatasetOperationService) CreateDatasetItemWithImages(ctx context.Context, req CreateDatasetItemWithImagesRequest) (*vo.ImageItemVO, error) {
	datasetRepo := dos.getDatasetRepo()

	// 1. 校验数据集是否存在
	_, err := datasetRepo.FindByID(ctx, req.DatasetID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, fmt.Errorf("数据集不存在")
		}
		return nil, fmt.Errorf("查询数据集失败: %w", err)
	}

	// 2. 校验清晰图信息
	if req.ClearImage.Path == "" {
		return nil, fmt.Errorf("清晰图路径不能为空")
	}

	// 3. 图片分辨率校验
	if req.Options.ValidateResolution {
		clearPath := req.ClearImage.Path
		hazyPaths := make([]string, 0, len(req.HazyImages))
		for _, img := range req.HazyImages {
			hazyPaths = append(hazyPaths, img.Path)
		}
		if err := dos.domain.ValidateResolution(clearPath, hazyPaths); err != nil {
			return nil, fmt.Errorf("图片分辨率校验失败: %w", err)
		}
	}

	// 开始事务
	tx := global.DB.Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	// 4. 创建数据集项
	itemName := req.ItemName
	if itemName == "" {
		itemName = fmt.Sprintf("Item_%d", time.Now().Unix())
	}

	datasetItem := model.SysDatasetItem{
		DatasetID: req.DatasetID,
		Name:      itemName,
	}

	if err := tx.Create(&datasetItem).Error; err != nil {
		tx.Rollback()
		return nil, fmt.Errorf("创建数据项失败: %w", err)
	}

	itemID := datasetItem.ID

	// 5. 保存清晰图文件
	clearFileID, err := dos.saveImageFile(ctx, tx, itemID, req.ClearImage, "clear", "")
	if err != nil {
		tx.Rollback()
		return nil, fmt.Errorf("保存清晰图失败: %w", err)
	}

	// 6. 保存雾霾图文件
	hazyFileIDs := make([]int64, 0, len(req.HazyImages))
	for _, hazyImg := range req.HazyImages {
		hazeLevel := hazyImg.HazeLevel
		if hazeLevel == "" {
			hazeLevel = "medium" // 默认中度雾霾
		}
		fileID, err := dos.saveImageFile(ctx, tx, itemID, hazyImg, "hazy", hazeLevel)
		if err != nil {
			tx.Rollback()
			return nil, fmt.Errorf("保存雾霾图失败: %w", err)
		}
		hazyFileIDs = append(hazyFileIDs, fileID)
	}

	// 7. 提交事务
	if err := tx.Commit().Error; err != nil {
		return nil, fmt.Errorf("提交事务失败: %w", err)
	}

	// 8. 异步生成缩略图（使用事件机制）
	if !req.Options.SkipThumbnail {
		dos.submitThumbnailGenerationTask(ctx, req.DatasetID, itemID,
			append([]int64{clearFileID}, hazyFileIDs...))
	}

	// 9. 失效统计缓存
	dos.invalidateDatasetStatsCache(req.DatasetID)

	logger.Info("创建数据项成功",
		zap.Int64("datasetID", req.DatasetID),
		zap.Int64("itemID", itemID),
		zap.Int("clearFiles", 1),
		zap.Int("hazyFiles", len(req.HazyImages)))

	// 查询创建的文件URL列表
	itemFileRepo := dos.getItemFileRepo()
	itemFiles, err := itemFileRepo.FindByItemID(ctx, itemID)
	if err != nil {
		logger.Warn("查询项文件失败", zap.Error(err))
		return &vo.ImageItemVO{
			ID:         itemID,
			DatasetID:  req.DatasetID,
			Name:       itemName,
			ImageCount: 0,
			HazyImages: []vo.ImageUrlVO{},
			CreateTime: time.Now().Format("2006-01-02 15:04:05"),
			UpdateTime: time.Now().Format("2006-01-02 15:04:05"),
		}, nil
	}

	imageUrls := make([]vo.ImageUrlVO, 0, len(itemFiles))
	for _, itemFile := range itemFiles {
		var file model.SysFile
		if err := global.DB.Where("id = ?", itemFile.FileID).First(&file).Error; err == nil {
			imageUrls = append(imageUrls, vo.ImageUrlVO{
				ID:          itemFile.ID,
				Type:        itemFile.Type,
				URL:         utils.StringVal(file.URL),
				OriginURL:   utils.StringVal(file.URL),
				Description: utils.StringVal(itemFile.Description),
			})
		}
	}

	return &vo.ImageItemVO{
		ID:         itemID,
		DatasetID:  req.DatasetID,
		Name:       itemName,
		ImageCount: len(imageUrls),
		HazyImages: imageUrls,
		CreateTime: time.Now().Format("2006-01-02 15:04:05"),
		UpdateTime: time.Now().Format("2006-01-02 15:04:05"),
	}, nil
}

// saveImageFile 保存图片文件到数据库
func (dos *DatasetOperationService) saveImageFile(ctx context.Context, tx *gorm.DB, itemID int64, img ImageUploadInfo, imgType, hazeLevel string) (int64, error) {
	// 创建或获取文件记录
	fileBO := bo.FileBO{
		Name:       img.Name,
		ObjectName: img.Name,
		Extension:  dos.getExtension(img.Name),
		MD5:        img.MD5,
		Path:       img.Path,
		Size:       img.Size,
		URL:        img.URL,
	}

	// 保存文件记录
	file := model.SysFile{
		Name:       fileBO.Name,
		ObjectName: fileBO.ObjectName,
		Path:       fileBO.Path,
		Size:       formatSize(fileBO.Size),
		MD5:        fileBO.MD5,
		URL:        &fileBO.URL,
	}

	if err := tx.Create(&file).Error; err != nil {
		return 0, fmt.Errorf("创建文件记录失败: %w", err)
	}

	// 创建项文件关联记录
	itemFile := model.SysItemFile{
		ItemID: itemID,
		FileID: int64(file.ID),
		Type:   imgType,
	}

	// 如果是雾霾图，设置雾霾程度描述
	if hazeLevel != "" {
		hazeDesc := fmt.Sprintf("雾霾程度: %s", hazeLevel)
		itemFile.Description = &hazeDesc
	}

	if err := tx.Create(&itemFile).Error; err != nil {
		return 0, fmt.Errorf("创建项文件关联失败: %w", err)
	}

	return int64(file.ID), nil
}

// BatchCreateDatasetItemsWithImagesRequest 批量创建请求
type BatchCreateDatasetItemsWithImagesRequest struct {
	DatasetID int64              `json:"datasetId" validate:"required"`
	Items     []BatchItemRequest `json:"items"`
	Options   CreateItemOptions  `json:"options"`
}

// BatchItemRequest 批量项请求
type BatchItemRequest struct {
	Prefix     string            `json:"prefix"`   // 前缀，用于分组
	ItemName   string            `json:"itemName"` // 数据项名称
	SceneType  string            `json:"sceneType"`
	ClearImage ImageUploadInfo   `json:"clearImage"`
	HazyImages []ImageUploadInfo `json:"hazyImages"`
}

// BatchCreateDatasetItemsWithImages 批量创建数据项并上传配对图片
// 支持前缀分组，相同前缀的项会组织在一起
func (dos *DatasetOperationService) BatchCreateDatasetItemsWithImages(ctx context.Context, req BatchCreateDatasetItemsWithImagesRequest) (*BatchCreateResult, error) {
	datasetRepo := dos.getDatasetRepo()

	// 1. 校验数据集
	_, err := datasetRepo.FindByID(ctx, req.DatasetID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, fmt.Errorf("数据集不存在")
		}
		return nil, fmt.Errorf("查询数据集失败: %w", err)
	}

	if len(req.Items) == 0 {
		return nil, fmt.Errorf("批量创建列表不能为空")
	}

	// 2. 批量创建数据项
	result := &BatchCreateResult{
		Total:      len(req.Items),
		Success:    0,
		Failed:     0,
		ItemIDs:    make([]int64, 0),
		Errors:     make([]BatchItemError, 0),
		GroupItems: make(map[string][]int64),
	}

	for i, itemReq := range req.Items {
		itemReq := itemReq
		createReq := CreateDatasetItemWithImagesRequest{
			DatasetID:  req.DatasetID,
			ItemName:   itemReq.ItemName,
			SceneType:  itemReq.SceneType,
			ClearImage: itemReq.ClearImage,
			HazyImages: itemReq.HazyImages,
			Options:    req.Options,
		}

		// 如果没有指定名称，使用前缀生成
		if createReq.ItemName == "" {
			createReq.ItemName = fmt.Sprintf("%s_%d", itemReq.Prefix, i+1)
		}

		itemVO, err := dos.CreateDatasetItemWithImages(ctx, createReq)
		if err != nil {
			result.Failed++
			result.Errors = append(result.Errors, BatchItemError{
				Index:    i,
				ItemName: createReq.ItemName,
				Error:    err.Error(),
			})
			logger.Warn("批量创建数据项失败",
				zap.Int("index", i),
				zap.String("itemName", createReq.ItemName),
				zap.Error(err))
			continue
		}

		result.Success++
		result.ItemIDs = append(result.ItemIDs, itemVO.ID)

		// 按前缀分组
		prefix := itemReq.Prefix
		if prefix == "" {
			prefix = "default"
		}
		result.GroupItems[prefix] = append(result.GroupItems[prefix], itemVO.ID)
	}

	logger.Info("批量创建数据项完成",
		zap.Int64("datasetID", req.DatasetID),
		zap.Int("total", result.Total),
		zap.Int("success", result.Success),
		zap.Int("failed", result.Failed))

	return result, nil
}

// BatchCreateResult 批量创建结果
type BatchCreateResult struct {
	Total      int                `json:"total"`
	Success    int                `json:"success"`
	Failed     int                `json:"failed"`
	ItemIDs    []int64            `json:"itemIds"`
	Errors     []BatchItemError   `json:"errors,omitempty"`
	GroupItems map[string][]int64 `json:"groupItems,omitempty"` // 按前缀分组的结果
}

// BatchItemError 批量创建错误
type BatchItemError struct {
	Index    int    `json:"index"`
	ItemName string `json:"itemName"`
	Error    string `json:"error"`
}

// DeleteDatasetItemCascade 级联删除数据项
// 删除数据项及其关联的文件记录和物理文件
func (dos *DatasetOperationService) DeleteDatasetItemCascade(ctx context.Context, itemID int64) error {
	datasetItemRepo := dos.getDatasetItemRepo()
	itemFileRepo := dos.getItemFileRepo()

	// 1. 查询数据项
	item, err := datasetItemRepo.FindByID(ctx, itemID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return fmt.Errorf("数据项不存在")
		}
		return fmt.Errorf("查询数据项失败: %w", err)
	}

	datasetID := item.DatasetID

	// 2. 查询所有关联的文件
	itemFiles, err := itemFileRepo.FindByItemID(ctx, itemID)
	if err != nil {
		return fmt.Errorf("查询项文件失败: %w", err)
	}

	// 收集需要删除的物理文件路径
	var filePaths []string
	fileIDs := make([]int64, 0, len(itemFiles))

	for _, itemFile := range itemFiles {
		fileIDs = append(fileIDs, itemFile.FileID)

		// 查询文件信息
		var file model.SysFile
		err := global.DB.Where("id = ?", itemFile.FileID).First(&file).Error
		if err == nil && file.Path != "" {
			filePaths = append(filePaths, file.Path)
			// 如果有缩略图，也收集缩略图路径
			if itemFile.ThumbnailFileID != nil {
				var thumbFile model.SysFile
				if thumbErr := global.DB.Where("id = ?", *itemFile.ThumbnailFileID).First(&thumbFile).Error; thumbErr == nil {
					filePaths = append(filePaths, thumbFile.Path)
				}
			}
		}
	}

	// 3. 开启事务
	tx := global.DB.Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	// 4. 删除项文件关联记录
	if err := tx.Where("item_id = ?", itemID).Delete(&model.SysItemFile{}).Error; err != nil {
		tx.Rollback()
		return fmt.Errorf("删除项文件关联失败: %w", err)
	}

	// 5. 删除数据项
	if err := tx.Delete(&model.SysDatasetItem{}, itemID).Error; err != nil {
		tx.Rollback()
		return fmt.Errorf("删除数据项失败: %w", err)
	}

	// 6. 提交事务
	if err := tx.Commit().Error; err != nil {
		return fmt.Errorf("提交事务失败: %w", err)
	}

	// 7. 异步删除物理文件
	if len(filePaths) > 0 {
		dos.submitFileDeletionTask(ctx, datasetID, filePaths)
	}

	// 8. 失效统计缓存
	dos.invalidateDatasetStatsCache(datasetID)

	logger.Info("级联删除数据项成功",
		zap.Int64("itemID", itemID),
		zap.Int64("datasetID", datasetID),
		zap.Int("files", len(filePaths)))

	return nil
}

// BatchDeleteDatasets 批量删除数据集
// 支持级联删除子数据集、数据项和文件
func (dos *DatasetOperationService) BatchDeleteDatasets(ctx context.Context, req bo.BatchDeleteForm) (*BatchCreateResult, error) {
	datasetItemRepo := dos.getDatasetItemRepo()

	if len(req.IDs) == 0 {
		return nil, fmt.Errorf("删除ID列表不能为空")
	}

	// 1. 查询所有需要删除的数据集（包括子数据集）
	// 注意：Repository 接口暂无 FindAll 方法，暂用 global.DB
	var allDatasets []model.SysDataset
	err := global.DB.Where("deleted = ?", 0).Find(&allDatasets).Error
	if err != nil {
		return nil, fmt.Errorf("查询数据集失败: %w", err)
	}

	// 转换为树节点
	nodes := make([]utils.TreeDataNode, 0, len(allDatasets))
	for i := range allDatasets {
		nodes = append(nodes, &allDatasets[i])
	}

	// 收集所有需要删除的数据集ID（包括子数据集）
	allDeleteIDs := make(map[int64]bool)
	for _, id := range req.IDs {
		allDeleteIDs[id] = true
		// 递归获取子节点
		children := dos.treeUtils.GetDescendantIDs(nodes, id)
		for _, childID := range children {
			allDeleteIDs[childID] = true
		}
	}

	idsToDelete := make([]int64, 0, len(allDeleteIDs))
	for id := range allDeleteIDs {
		idsToDelete = append(idsToDelete, id)
	}

	// 2. 如果不强制删除，检查是否有数据项
	if !req.Force {
		// 统计数据项数量
		totalCount := int64(0)
		for _, datasetID := range idsToDelete {
			items, err := datasetItemRepo.FindByDatasetID(ctx, datasetID)
			if err != nil {
				return nil, fmt.Errorf("检查数据项失败: %w", err)
			}
			totalCount += int64(len(items))
		}
		if totalCount > 0 {
			return nil, fmt.Errorf("数据集下还有 %d 个数据项，请使用 force=true 强制删除", totalCount)
		}
	}

	// 3. 查询所有关联的文件
	var allDatasetItems []model.SysDatasetItem
	for _, datasetID := range idsToDelete {
		items, err := datasetItemRepo.FindByDatasetID(ctx, datasetID)
		if err != nil {
			return nil, fmt.Errorf("查询数据项失败: %w", err)
		}
		allDatasetItems = append(allDatasetItems, items...)
	}

	itemIDs := make([]int64, 0, len(allDatasetItems))
	for _, item := range allDatasetItems {
		itemIDs = append(itemIDs, item.ID)
	}

	// 查询所有文件记录
	var allItemFiles []model.SysItemFile
	fileIDs := make([]int64, 0)
	if len(itemIDs) > 0 {
		for _, itemID := range itemIDs {
			var itemFiles []model.SysItemFile
			if err := global.DB.Where("item_id = ?", itemID).Find(&itemFiles).Error; err != nil {
				return nil, fmt.Errorf("查询项文件失败: %w", err)
			}
			for _, itemFile := range itemFiles {
				allItemFiles = append(allItemFiles, itemFile)
				fileIDs = append(fileIDs, itemFile.FileID)
			}
		}
	}

	// 收集物理文件路径
	filePathsMap := make(map[int64]string)
	for _, fileID := range fileIDs {
		var file model.SysFile
		err := global.DB.Where("id = ?", fileID).First(&file).Error
		if err == nil && file.Path != "" {
			filePathsMap[fileID] = file.Path
		}
	}

	// 4. 开启事务
	tx := global.DB.Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	// 5. 删除文件关联记录
	if len(itemIDs) > 0 {
		if err := tx.Where("item_id IN ?", itemIDs).Delete(&model.SysItemFile{}).Error; err != nil {
			tx.Rollback()
			return nil, fmt.Errorf("删除项文件关联失败: %w", err)
		}
	}

	// 6. 删除数据项
	if len(itemIDs) > 0 {
		if err := tx.Where("id IN ?", itemIDs).Delete(&model.SysDatasetItem{}).Error; err != nil {
			tx.Rollback()
			return nil, fmt.Errorf("删除数据项失败: %w", err)
		}
	}

	// 7. 删除数据集（逻辑删除）
	// 修复 P0 级别批量删除逻辑错误：确保批量更新时正确填充 update_by 字段
	if err := tx.Where("id IN ?", idsToDelete).
		Updates(map[string]interface{}{
			"deleted":     1,
			"update_time": time.Now(),
			"update_by":   getBatchUpdateUserID(ctx), // 获取当前用户ID并填充到 update_by 字段
		}).Error; err != nil {
		tx.Rollback()
		return nil, fmt.Errorf("删除数据集失败: %w", err)
	}

	// 8. 提交事务
	if err := tx.Commit().Error; err != nil {
		return nil, fmt.Errorf("提交事务失败: %w", err)
	}

	// 9. 异步删除物理文件
	filePaths := make([]string, 0, len(filePathsMap))
	for _, path := range filePathsMap {
		filePaths = append(filePaths, path)
	}
	if len(filePaths) > 0 {
		dos.submitFileDeletionTask(ctx, 0, filePaths)
	}

	// 10. 清理缓存
	dos.invalidateTreeCache()
	for _, datasetID := range idsToDelete {
		dos.invalidateDatasetStatsCache(datasetID)
	}

	logger.Info("批量删除数据集成功",
		zap.Int("requested", len(req.IDs)),
		zap.Int("deleted", len(idsToDelete)),
		zap.Int("items", len(itemIDs)),
		zap.Int("files", len(fileIDs)))

	return &BatchCreateResult{
		Total:   len(req.IDs),
		Success: len(idsToDelete),
		Failed:  0,
		ItemIDs: idsToDelete,
	}, nil
}

// ========== 异步任务相关 ==========

// submitThumbnailGenerationTask 提交缩略图生成任务
func (dos *DatasetOperationService) submitThumbnailGenerationTask(ctx context.Context, datasetID, itemID int64, fileIDs []int64) {
	taskIDStr := fmt.Sprintf("thumb_%d_%d", datasetID, itemID)

	pool := dos.taskExecutor.getWorkerPool("thumbnail")
	if pool == nil {
		logger.Error("缩略图Worker池不存在")
		return
	}

	pool.SubmitWithCtx(taskIDStr, "thumbnail", len(fileIDs), func(ctx context.Context) error {
		logger.Info("开始生成缩略图",
			zap.Int64("datasetID", datasetID),
			zap.Int64("itemID", itemID),
			zap.Int("fileCount", len(fileIDs)))

		for i, fileID := range fileIDs {
			// 检查是否取消
			if dos.isTaskCanceled(ctx, taskIDStr) {
				return errors.New("任务被取消")
			}

			if err := dos.generateThumbnail(fileID); err != nil {
				logger.Error("生成缩略图失败",
					zap.Int64("fileID", fileID),
					zap.Error(err))
				// 记录失败
				dos.recordThumbnailFailure(fileID)
			} else {
				// 更新进度
				if tc, ok := pool.GetTaskContext(taskIDStr); ok {
					tc.UpdateProgress(i + 1)
				}
			}
		}
		return nil
	})
}

// submitFileDeletionTask 提交文件删除任务
func (dos *DatasetOperationService) submitFileDeletionTask(ctx context.Context, datasetID int64, filePaths []string) {
	taskIDStr := fmt.Sprintf("delete_%d_%d", datasetID, time.Now().UnixNano())

	pool := dos.taskExecutor.getWorkerPool("dataset")
	if pool == nil {
		logger.Error("数据集Worker池不存在")
		return
	}

	pool.SubmitWithCtx(taskIDStr, "delete", len(filePaths), func(ctx context.Context) error {
		logger.Info("开始删除物理文件",
			zap.Int64("datasetID", datasetID),
			zap.Int("fileCount", len(filePaths)))

		for i, filePath := range filePaths {
			// 检查是否取消
			if dos.isTaskCanceled(ctx, taskIDStr) {
				return errors.New("任务被取消")
			}

			if err := dos.deletePhysicalFile(filePath); err != nil {
				logger.Error("删除文件失败",
					zap.String("path", filePath),
					zap.Error(err))
				// 记录失败
				dos.recordDeletionFailure(filePath)
			} else {
				// 更新进度
				if tc, ok := pool.GetTaskContext(taskIDStr); ok {
					tc.UpdateProgress(i + 1)
				}
			}
		}
		return nil
	})
}

// generateThumbnail 生成缩略图
func (dos *DatasetOperationService) generateThumbnail(fileID int64) error {
	// TODO: 实现缩略图生成逻辑
	// 1. 查询文件信息
	// 2. 生成缩略图
	// 3. 保存缩略图文件
	// 4. 更新 SysItemFile 的 thumbnail_file_id

	logger.Info("生成缩略图", zap.Int64("fileID", fileID))
	return nil
}

// deletePhysicalFile 删除物理文件
func (dos *DatasetOperationService) deletePhysicalFile(filePath string) error {
	// TODO: 实现文件删除逻辑
	// 1. 调用文件存储服务删除文件
	// 2. 同时删除缩略图

	logger.Info("删除物理文件", zap.String("path", filePath))
	return nil
}

// ========== 缓存相关 ==========

// invalidateDatasetStatsCache 失效数据集统计缓存
func (dos *DatasetOperationService) invalidateDatasetStatsCache(datasetID int64) {
	if global.REDIS == nil {
		return
	}
	ctx := context.Background()
	cacheKey := "dataset:stats:" + fmt.Sprintf("%d", datasetID)
	if err := global.REDIS.Del(ctx, cacheKey).Err(); err != nil {
		logger.Warn("失效统计缓存失败", zap.String("key", cacheKey), zap.Error(err))
	}
}

// invalidateTreeCache 失效树形结构缓存
func (dos *DatasetOperationService) invalidateTreeCache() {
	if global.REDIS == nil {
		return
	}
	ctx := context.Background()
	cacheKey := "dataset:tree"
	if err := global.REDIS.Del(ctx, cacheKey).Err(); err != nil {
		logger.Warn("失效树缓存失败", zap.String("key", cacheKey), zap.Error(err))
	}
}

// recordThumbnailFailure 记录缩略图生成失败
func (dos *DatasetOperationService) recordThumbnailFailure(fileID int64) {
	if global.REDIS == nil {
		return
	}
	ctx := context.Background()
	global.REDIS.HSet(ctx, "dataset:thumbnail:failed", fileID, time.Now().Unix())
}

// recordDeletionFailure 记录文件删除失败
func (dos *DatasetOperationService) recordDeletionFailure(filePath string) {
	if global.REDIS == nil {
		return
	}
	ctx := context.Background()
	global.REDIS.SAdd(ctx, "dataset:deletion:failed", filePath)
}

// isTaskCanceled 检查任务是否被取消
func (dos *DatasetOperationService) isTaskCanceled(ctx context.Context, taskID string) bool {
	select {
	case <-ctx.Done():
		return true
	default:
		// 检查Redis中的取消标志
		if global.REDIS == nil {
			return false
		}
		cancelKey := "task:cancel:" + taskID
		canceled, _ := global.REDIS.Get(ctx, cancelKey).Result()
		return canceled == "true"
	}
}

// ========== 辅助函数 ==========

// formatSize 格式化文件大小
func formatSize(size int64) string {
	if size < 1024 {
		return fmt.Sprintf("%d B", size)
	} else if size < 1024*1024 {
		return fmt.Sprintf("%.2f KB", float64(size)/1024)
	} else if size < 1024*1024*1024 {
		return fmt.Sprintf("%.2f MB", float64(size)/(1024*1024))
	}
	return fmt.Sprintf("%.2f GB", float64(size)/(1024*1024*1024))
}

// getExtension 获取文件扩展名
func (dos *DatasetOperationService) getExtension(filename string) string {
	idx := strings.LastIndex(filename, ".")
	if idx == -1 {
		return ""
	}
	return filename[idx:]
}

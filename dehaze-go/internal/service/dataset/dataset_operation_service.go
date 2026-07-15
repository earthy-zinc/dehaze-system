package dataset

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	datasetrepo "github.com/earthyzinc/dehaze-go/internal/repository/dataset"
	filerepo "github.com/earthyzinc/dehaze-go/internal/repository/file"
	fileservice "github.com/earthyzinc/dehaze-go/internal/service/file"
	taskservice "github.com/earthyzinc/dehaze-go/internal/service/task"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/database"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

// DatasetOperationService 数据集操作服务
// 负责处理复杂的数据集操作，如批量上传、级联删除等
// 注意：保留 *gorm.DB 用于复杂事务操作，其他操作通过 Repository 接口
type DatasetOperationService struct {
	cache types.ICache

	datasetRepo          datasetrepo.IDatasetRepository
	datasetItemRepo      datasetrepo.IDatasetItemRepository
	datasetItemFileRepo  datasetrepo.IDatasetItemFileRepository
	itemFileRepo         filerepo.IItemFileRepository
	fileRepo             filerepo.IFileRepository
	taskExecutor         taskservice.AsyncTaskExecutor
	pairedImageValidator *PairedImageValidator
	treeUtils            *utils.TreeDataUtils
}

// NewDatasetOperationService 创建数据集操作服务
func NewDatasetOperationService(
	cache types.ICache,
	datasetRepo datasetrepo.IDatasetRepository,
	datasetItemRepo datasetrepo.IDatasetItemRepository,
	datasetItemFileRepo datasetrepo.IDatasetItemFileRepository,
	itemFileRepo filerepo.IItemFileRepository,
	fileRepo filerepo.IFileRepository,
	taskExecutor taskservice.AsyncTaskExecutor,
) *DatasetOperationService {
	return &DatasetOperationService{
		cache:                cache,
		datasetRepo:          datasetRepo,
		datasetItemRepo:      datasetItemRepo,
		datasetItemFileRepo:  datasetItemFileRepo,
		itemFileRepo:         itemFileRepo,
		fileRepo:             fileRepo,
		taskExecutor:         taskExecutor,
		pairedImageValidator: NewPairedImageValidator(),
		treeUtils:            utils.NewTreeDataUtils(),
	}
}

// getBatchUpdateUserID 从上下文中获取当前用户ID
// 用于批量更新操作时的 update_by 字段填充
func getBatchUpdateUserID(ctx context.Context) int64 {
	if ctx == nil {
		return 0
	}
	return database.GetUserID(ctx)
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
func (dos *DatasetOperationService) CreateDatasetItemWithImages(
	ctx context.Context,
	req CreateDatasetItemWithImagesRequest,
) (*vo.ImageItemVO, error) {
	// 1. 校验数据集是否存在
	dataset, err := dos.datasetRepo.FindByID(ctx, req.DatasetID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "数据集不存在")
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询数据集失败", err)
	}
	if dataset == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "数据集不存在")
	}

	// 2. 清晰图和有雾图均为可选（适配不同数据集规范：GT+Hazy 配对型、仅 Hazy 无 GT 型等）
	if req.ClearImage.Path == "" && len(req.HazyImages) == 0 {
		return nil, common.NewBizError(common.PARAM_ERROR, "至少上传一张图片（清晰图或有雾图）")
	}

	// 3. 图片分辨率校验（仅在上传了清晰图时校验）
	if req.Options.ValidateResolution && req.ClearImage.Path != "" {
		clearPath := req.ClearImage.Path
		hazyPaths := make([]string, 0, len(req.HazyImages))
		for _, img := range req.HazyImages {
			hazyPaths = append(hazyPaths, img.Path)
		}
		if err := dos.pairedImageValidator.ValidateResolution(clearPath, hazyPaths); err != nil {
			return nil, common.WrapBizError(common.PARAM_ERROR, "图片分辨率校验失败", err)
		}
	}

	// 4. 创建数据集项及关联文件（事务下沉到 Repository）
	itemName := req.ItemName
	if itemName == "" {
		itemName = fmt.Sprintf("Item_%d", time.Now().Unix())
	}

	itemFiles := make([]datasetrepo.ItemFileCreate, 0, 1+len(req.HazyImages))
	// 清晰图（可选）
	if req.ClearImage.Path != "" {
		clearFileType := dos.getExtension(req.ClearImage.Name)
		if clearFileType == "" {
			clearFileType = dos.getExtension(req.ClearImage.Path)
		}
		itemFiles = append(itemFiles, datasetrepo.ItemFileCreate{
			Type:      "clear",
			Name:      req.ClearImage.Name,
			Path:      req.ClearImage.Path,
			URL:       req.ClearImage.URL,
			Size:      req.ClearImage.Size,
			MD5:       req.ClearImage.MD5,
			FileType:  clearFileType,
			SceneType: req.SceneType,
		})
	}

	// 有雾图（可选，haze_level 支持多种规范：light/medium/heavy、beta=X、A=X,beta=Y 等，可为空）
	for _, hazyImg := range req.HazyImages {
		fileType := dos.getExtension(hazyImg.Name)
		if fileType == "" {
			fileType = dos.getExtension(hazyImg.Path)
		}
		itemFiles = append(itemFiles, datasetrepo.ItemFileCreate{
			Type:      "hazy",
			Name:      hazyImg.Name,
			Path:      hazyImg.Path,
			URL:       hazyImg.URL,
			Size:      hazyImg.Size,
			MD5:       hazyImg.MD5,
			HazeLevel: hazyImg.HazeLevel, // 可为空，不再默认填充 "medium"
			FileType:  fileType,
			SceneType: req.SceneType,
		})
	}

	itemID, fileIDs, err := dos.datasetItemFileRepo.CreateDatasetItemWithFiles(ctx, req.DatasetID, itemName, itemFiles)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建数据项及关联文件失败", err)
	}

	// 8. 生成缩略图
	if !req.Options.SkipThumbnail {
		if req.Options.AsyncThumbnail {
			dos.submitThumbnailGenerationTask(ctx, req.DatasetID, itemID, fileIDs)
		} else {
			for _, fileID := range fileIDs {
				if err := dos.generateThumbnail(fileID); err != nil {
					logger.Error("生成缩略图失败", zap.Int64("fileID", fileID), zap.Error(err))
				}
			}
		}
	}

	// 9. 失效统计与列表缓存
	dos.invalidateDatasetStatsCache(ctx, req.DatasetID)
	dos.invalidateDatasetItemsCache(ctx, req.DatasetID)

	logger.Info("创建数据项成功",
		zap.Int64("datasetID", req.DatasetID),
		zap.Int64("itemID", itemID),
		zap.Int("clearFiles", 1),
		zap.Int("hazyFiles", len(req.HazyImages)))

	// 查询创建的文件URL列表
	createdItemFiles, err := dos.itemFileRepo.FindByItemID(ctx, itemID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询创建的项文件失败", err)
	}

	// 收集文件ID
	createdFileIDs := make([]int64, 0, len(createdItemFiles))
	for _, itemFile := range createdItemFiles {
		createdFileIDs = append(createdFileIDs, itemFile.FileID)
	}

	// 查询文件信息
	files, err := dos.fileRepo.FindByIDs(ctx, createdFileIDs)
	if err != nil {
		logger.Warn("查询文件失败", zap.Error(err))
	}

	// 构建文件信息映射
	fileURLMap := make(map[int64]string)
	fileInfoMap := make(map[int64]*model.SysFile)
	for i := range files {
		fileURLMap[int64(files[i].ID)] = utils.StringVal(files[i].URL)
		fileInfoMap[int64(files[i].ID)] = &files[i]
	}

	imageUrls := make([]vo.ImageUrlVO, 0, len(createdItemFiles))
	var clearImage *vo.ImageUrlVO
	var sceneTypeStr, descriptionStr string
	for _, itemFile := range createdItemFiles {
		url := fileURLMap[itemFile.FileID]
		fileInfo := fileInfoMap[itemFile.FileID]

		imageUrlVO := fileservice.BuildImageUrlVO(fileInfo, &itemFile, url)
		imageUrlVO.ID = itemFile.ID
		imageUrlVO.ItemID = itemFile.ItemID
		imageUrlVO.DatasetID = req.DatasetID

		if itemFile.Type == "clear" {
			clearImage = &imageUrlVO
		} else {
			imageUrls = append(imageUrls, imageUrlVO)
		}

		if sceneTypeStr == "" && itemFile.SceneType != nil {
			sceneTypeStr = utils.StringVal(itemFile.SceneType)
		}
		if descriptionStr == "" && itemFile.Description != nil {
			descriptionStr = utils.StringVal(itemFile.Description)
		}
	}

	imageCount := len(imageUrls)
	if clearImage != nil {
		imageCount++
	}

	return &vo.ImageItemVO{
		ID:          itemID,
		DatasetID:   req.DatasetID,
		Name:        itemName,
		SceneType:   sceneTypeStr,
		Description: descriptionStr,
		ImageCount:  imageCount,
		ClearImage:  clearImage,
		HazyImages:  imageUrls,
		CreateTime:  time.Now().Format("2006-01-02 15:04:05"),
		UpdateTime:  time.Now().Format("2006-01-02 15:04:05"),
	}, nil
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
	// 1. 校验数据集
	dataset, err := dos.datasetRepo.FindByID(ctx, req.DatasetID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "数据集不存在")
		}
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询数据集失败", err)
	}
	if dataset == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "数据集不存在")
	}

	if len(req.Items) == 0 {
		return nil, common.NewBizError(common.PARAM_ERROR, "批量创建列表不能为空")
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

// BatchDeleteResult 批量删除结果
type BatchDeleteResult struct {
	Total      int     `json:"total"`      // 请求删除的数据集数量
	Deleted    int     `json:"deleted"`    // 实际删除的数据集数量（含子数据集）
	DatasetIDs []int64 `json:"datasetIds"` // 被删除的数据集ID列表
	ItemCount  int     `json:"itemCount"`  // 关联删除的数据项数量
	FileCount  int     `json:"fileCount"`  // 关联删除的文件数量
}

// DeleteDatasetItemCascade 级联删除数据项
// 删除数据项及其关联的文件记录和物理文件
func (dos *DatasetOperationService) DeleteDatasetItemCascade(ctx context.Context, itemID int64) error {
	// 1. 查询数据项
	item, err := dos.datasetItemRepo.FindByID(ctx, itemID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return common.NewBizError(common.RESOURCE_NOT_FOUND, "数据项不存在")
		}
		return common.WrapBizError(common.DATABASE_ERROR, "查询数据项失败", err)
	}
	if item == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "数据项不存在")
	}

	datasetID := item.DatasetID

	// 2. 查询所有关联的文件
	itemFiles, err := dos.itemFileRepo.FindByItemID(ctx, itemID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询项文件失败", err)
	}

	// 收集需要删除的物理文件路径
	fileIDSet := make(map[int64]struct{})
	for _, itemFile := range itemFiles {
		fileIDSet[itemFile.FileID] = struct{}{}
		if itemFile.ThumbnailFileID != nil {
			fileIDSet[*itemFile.ThumbnailFileID] = struct{}{}
		}
	}

	fileIDs := make([]int64, 0, len(fileIDSet))
	for id := range fileIDSet {
		fileIDs = append(fileIDs, id)
	}

	filePaths := make([]string, 0, len(fileIDs))
	if len(fileIDs) > 0 {
		files, err := dos.fileRepo.FindByIDs(ctx, fileIDs)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询文件失败", err)
		}
		for _, file := range files {
			if file.Path != "" {
				filePaths = append(filePaths, file.Path)
			}
		}
	}

	// 3. 级联删除项文件与数据项（事务下沉到 Repository）
	if err := dos.datasetItemFileRepo.DeleteDatasetItemCascade(ctx, itemID); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "级联删除数据项失败", err)
	}

	// 7. 异步删除物理文件
	if len(filePaths) > 0 {
		dos.submitFileDeletionTask(ctx, datasetID, filePaths)
	}

	// 8. 失效统计与列表缓存
	dos.invalidateDatasetStatsCache(ctx, datasetID)
	dos.invalidateDatasetItemsCache(ctx, datasetID)
	dos.invalidateDatasetItemCache(ctx, itemID)

	logger.Info("级联删除数据项成功",
		zap.Int64("itemID", itemID),
		zap.Int64("datasetID", datasetID),
		zap.Int("files", len(filePaths)))

	return nil
}

// BatchDeleteDatasets 批量删除数据集
// 支持级联删除子数据集、数据项和文件
func (dos *DatasetOperationService) BatchDeleteDatasets(ctx context.Context, req bo.BatchDeleteForm) (*BatchDeleteResult, error) {
	if len(req.IDs) == 0 {
		return nil, common.NewBizError(common.PARAM_ERROR, "删除ID列表不能为空")
	}

	// 1. 查询所有数据集
	allDatasets, err := dos.datasetRepo.FindAll(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询数据集失败", err)
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
		totalCount, err := dos.datasetItemRepo.CountByDatasetIDs(ctx, idsToDelete)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "检查数据项失败", err)
		}
		if totalCount > 0 {
			return nil, common.NewBizError(common.BUSINESS_ERROR, fmt.Sprintf("数据集下还有 %d 个数据项，请使用 force=true 强制删除", totalCount))
		}
	}

	// 3. 查询所有关联的文件
	allDatasetItems, err := dos.datasetItemRepo.FindByDatasetIDs(ctx, idsToDelete)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询数据项失败", err)
	}

	itemIDs := make([]int64, 0, len(allDatasetItems))
	for _, item := range allDatasetItems {
		itemIDs = append(itemIDs, item.ID)
	}

	// 查询所有文件记录
	var allItemFiles []model.SysItemFile
	fileIDs := make([]int64, 0)
	if len(itemIDs) > 0 {
		allItemFiles, err = dos.itemFileRepo.FindByItemIDs(ctx, itemIDs)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询项文件失败", err)
		}
		fileIDSet := make(map[int64]struct{})
		for _, itemFile := range allItemFiles {
			fileIDSet[itemFile.FileID] = struct{}{}
			if itemFile.ThumbnailFileID != nil {
				fileIDSet[*itemFile.ThumbnailFileID] = struct{}{}
			}
		}
		for id := range fileIDSet {
			fileIDs = append(fileIDs, id)
		}
	}

	// 收集物理文件路径
	filePathsMap := make(map[int64]string)
	if len(fileIDs) > 0 {
		files, err := dos.fileRepo.FindByIDs(ctx, fileIDs)
		if err != nil {
			logger.Warn("查询文件失败", zap.Error(err))
		}
		for _, file := range files {
			if file.Path != "" {
				filePathsMap[int64(file.ID)] = file.Path
			}
		}
	}

	// 4. 级联删除数据项与关联记录（事务下沉到 Repository）
	if len(itemIDs) > 0 {
		if err := dos.datasetItemFileRepo.DeleteDatasetItemsCascade(ctx, itemIDs); err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "级联删除数据项失败", err)
		}
	}

	// 5. 删除数据集（逻辑删除）
	updateBy := getBatchUpdateUserID(ctx)
	if err := dos.datasetRepo.SoftDeleteByIDs(ctx, idsToDelete, updateBy); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "删除数据集失败", err)
	}

	// 6. 异步删除物理文件
	filePaths := make([]string, 0, len(filePathsMap))
	for _, path := range filePathsMap {
		filePaths = append(filePaths, path)
	}
	if len(filePaths) > 0 {
		dos.submitFileDeletionTask(ctx, 0, filePaths)
	}

	// 10. 清理缓存
	dos.invalidateTreeCache(ctx)
	for _, datasetID := range idsToDelete {
		dos.invalidateDatasetStatsCache(ctx, datasetID)
		dos.invalidateDatasetItemsCache(ctx, datasetID)
	}

	logger.Info("批量删除数据集成功",
		zap.Int("requested", len(req.IDs)),
		zap.Int("deleted", len(idsToDelete)),
		zap.Int("items", len(itemIDs)),
		zap.Int("files", len(fileIDs)))

	return &BatchDeleteResult{
		Total:      len(req.IDs),
		Deleted:    len(idsToDelete),
		DatasetIDs: idsToDelete,
		ItemCount:  len(itemIDs),
		FileCount:  len(fileIDs),
	}, nil
}

// ========== 异步任务相关 ==========

// submitThumbnailGenerationTask 提交缩略图生成任务
func (dos *DatasetOperationService) submitThumbnailGenerationTask(ctx context.Context, datasetID, itemID int64, fileIDs []int64) {
	taskIDStr := fmt.Sprintf("thumb_%d_%d", datasetID, itemID)

	if dos.taskExecutor == nil {
		logger.Error("任务执行器未初始化")
		return
	}

	payload := taskservice.ThumbnailBatchPayload{
		DatasetID: datasetID,
		ItemID:    itemID,
		FileIDs:   fileIDs,
	}
	msg := taskservice.TaskMessage{
		TaskID:    taskIDStr,
		TaskType:  "thumbnail",
		Total:     len(fileIDs),
		Payload:   payload,
		CreatedAt: time.Now(),
	}
	if err := dos.taskExecutor.PublishTask(ctx, msg); err != nil {
		logger.Error("提交缩略图任务失败", zap.String("taskID", taskIDStr), zap.Error(err))
	}
}

// submitFileDeletionTask 提交文件删除任务
func (dos *DatasetOperationService) submitFileDeletionTask(ctx context.Context, datasetID int64, filePaths []string) {
	taskIDStr := fmt.Sprintf("delete_%d_%d", datasetID, time.Now().UnixNano())

	if dos.taskExecutor == nil {
		logger.Error("任务执行器未初始化")
		return
	}

	payload := taskservice.FileDeletionPayload{
		DatasetID: datasetID,
		FilePaths: filePaths,
	}
	msg := taskservice.TaskMessage{
		TaskID:    taskIDStr,
		TaskType:  "dataset",
		Total:     len(filePaths),
		Payload:   payload,
		CreatedAt: time.Now(),
	}
	if err := dos.taskExecutor.PublishTask(ctx, msg); err != nil {
		logger.Error("提交文件删除任务失败", zap.String("taskID", taskIDStr), zap.Error(err))
	}
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

// ========== 缓存相关 ==========

// invalidateDatasetStatsCache 失效数据集统计缓存
func (dos *DatasetOperationService) invalidateDatasetStatsCache(ctx context.Context, datasetID int64) {
	if dos.cache == nil {
		return
	}
	keys := []string{
		fmt.Sprintf("dataset:stats:%d", datasetID),
		fmt.Sprintf("dataset:leaf:%d", datasetID),
		"dataset:all",
		"dataset:statsMap:all",
	}
	for _, key := range keys {
		if err := dos.cache.Delete(ctx, key); err != nil {
			logger.Warn("失效缓存失败", zap.String("key", key), zap.Error(err))
		}
	}
}

// invalidateDatasetItemsCache 失效数据集下所有数据项列表缓存
func (dos *DatasetOperationService) invalidateDatasetItemsCache(ctx context.Context, datasetID int64) {
	if dos.cache == nil {
		return
	}
	cacheKey := fmt.Sprintf("dataset:items:%d", datasetID)
	if err := dos.cache.Delete(ctx, cacheKey); err != nil {
		logger.Warn("失效数据项列表缓存失败", zap.String("key", cacheKey), zap.Error(err))
	}
}

// invalidateDatasetItemCache 失效单个数据项缓存
func (dos *DatasetOperationService) invalidateDatasetItemCache(ctx context.Context, itemID int64) {
	if dos.cache == nil {
		return
	}
	cacheKey := fmt.Sprintf("dataset:item:%d", itemID)
	if err := dos.cache.Delete(ctx, cacheKey); err != nil {
		logger.Warn("失效数据项缓存失败", zap.String("key", cacheKey), zap.Error(err))
	}
}

// invalidateTreeCache 失效树形结构缓存（含 options 子缓存）
func (dos *DatasetOperationService) invalidateTreeCache(ctx context.Context) {
	if dos.cache == nil {
		return
	}
	keys := []string{
		"dataset:tree",
		"dataset:tree:options",
		"dataset:all",
		"dataset:statsMap:all",
	}
	for _, key := range keys {
		if err := dos.cache.Delete(ctx, key); err != nil {
			logger.Warn("失效树缓存失败", zap.String("key", key), zap.Error(err))
		}
	}
}

// ========== 辅助函数 ==========

// getExtension 获取文件扩展名
func (dos *DatasetOperationService) getExtension(filename string) string {
	idx := strings.LastIndex(filename, ".")
	if idx == -1 {
		return ""
	}
	return filename[idx:]
}

package dataset

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	datasetrepo "github.com/earthyzinc/dehaze-go/internal/repository/dataset"
	filerepo "github.com/earthyzinc/dehaze-go/internal/repository/file"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"go.uber.org/zap"
)

const (
	// DATASET_STATS_TTL 统计缓存过期时间（30分钟）
	DATASET_STATS_TTL = 30 * time.Minute
	// DATASET_TREE_TTL 树形结构缓存过期时间（1小时）
	DATASET_TREE_TTL = time.Hour
	// DATASET_LEAF_TTL 叶子节点缓存过期时间（1小时）
	DATASET_LEAF_TTL = time.Hour
)

// DatasetService 数据集服务
type DatasetService struct {
	cache types.ICache

	datasetRepo     datasetrepo.IDatasetRepository
	datasetItemRepo datasetrepo.IDatasetItemRepository
	itemFileRepo    filerepo.IItemFileRepository
	fileRepo        filerepo.IFileRepository
	treeUtils       *utils.TreeDataUtils
}

// NewDatasetService 创建数据集服务实例
func NewDatasetService(cache types.ICache, datasetRepo datasetrepo.IDatasetRepository, datasetItemRepo datasetrepo.IDatasetItemRepository, itemFileRepo filerepo.IItemFileRepository, fileRepo filerepo.IFileRepository) *DatasetService {
	if datasetRepo == nil {
		panic("DatasetService: datasetRepo 未初始化")
	}
	if datasetItemRepo == nil {
		panic("DatasetService: datasetItemRepo 未初始化")
	}
	if itemFileRepo == nil {
		panic("DatasetService: itemFileRepo 未初始化")
	}
	if fileRepo == nil {
		panic("DatasetService: fileRepo 未初始化")
	}

	return &DatasetService{
		cache:           cache,
		datasetRepo:     datasetRepo,
		datasetItemRepo: datasetItemRepo,
		itemFileRepo:    itemFileRepo,
		fileRepo:        fileRepo,
		treeUtils:       utils.NewTreeDataUtils(),
	}
}

// SetDatasetRepo 设置 Repository（测试用）
func (s *DatasetService) SetDatasetRepo(repo datasetrepo.IDatasetRepository) {
	s.datasetRepo = repo
}

// DatasetStatistics 数据集统计信息
type DatasetStatistics struct {
	ItemCount          int64            `json:"itemCount"`
	FileCount          int64            `json:"fileCount"`
	TotalSize          int64            `json:"totalSize"`
	ClearCount         int64            `json:"clearCount"`
	HazyCount          int64            `json:"hazyCount"`
	SceneDistribution  map[string]int64 `json:"sceneDistribution"`
	HazeDistribution   map[string]int64 `json:"hazeDistribution"`
	FormatDistribution map[string]int64 `json:"formatDistribution"`
}

// ====================
// IDatasetService 接口实现
// ====================

// GetPage 数据集分页列表
func (datasetService *DatasetService) GetPage(ctx context.Context, q *query.DatasetQuery) (*vo.PageResult[vo.DatasetVO], error) {
	readResult, err := datasetService.datasetRepo.FindPage(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询数据集分页列表失败", err)
	}
	if readResult == nil {
		return &vo.PageResult[vo.DatasetVO]{List: []vo.DatasetVO{}, Total: 0}, nil
	}

	voList := make([]vo.DatasetVO, 0, len(readResult.List))
	for _, item := range readResult.List {
		voList = append(voList, mapDatasetReadToVO(item))
	}

	return &vo.PageResult[vo.DatasetVO]{
		List:     voList,
		Total:    readResult.Total,
		PageNum:  readResult.PageNum,
		PageSize: readResult.PageSize,
	}, nil
}

// GetDatasetOptions 数据集下拉选项
func (datasetService *DatasetService) GetDatasetOptions() (options []vo.Option, err error) {
	ctx := context.Background()
	cacheKey := CACHE_KEY_DATASET_TREE + ":options"

	if datasetService.cache != nil {
		cachedData, err := datasetService.cache.Get(ctx, cacheKey)
		if err == nil && cachedData != "" {
			if err := json.Unmarshal([]byte(cachedData), &options); err == nil {
				logger.Debug("下拉选项命中缓存")
				return options, nil
			}
		}
	}

	datasetList, err := datasetService.datasetRepo.FindAllActive(ctx)
	if err != nil {
		return options, common.WrapBizError(common.DATABASE_ERROR, "查询数据集列表失败", err)
	}

	if len(datasetList) == 0 {
		return options, nil
	}

	options = datasetService.buildTreeOptions(datasetList, 0)

	if datasetService.cache != nil {
		if optionsJSON, marshalErr := json.Marshal(options); marshalErr == nil {
			_ = datasetService.cache.Set(ctx, cacheKey, optionsJSON, DATASET_TREE_TTL)
		}
	}

	return options, nil
}

// buildTreeOptions 构建树形下拉选项
func (datasetService *DatasetService) buildTreeOptions(datasetList []model.SysDataset, rootID int64) []vo.Option {
	if len(datasetList) == 0 {
		return []vo.Option{}
	}

	// 构建父节点到子节点的映射
	parentToChildren := make(map[int64][]model.SysDataset)
	for _, dataset := range datasetList {
		parentID := dataset.ParentID
		if parentToChildren[parentID] == nil {
			parentToChildren[parentID] = []model.SysDataset{}
		}
		parentToChildren[parentID] = append(parentToChildren[parentID], dataset)
	}

	// 如果指定了rootID，从该根节点开始构建
	var roots []model.SysDataset
	if rootID != 0 {
		if children, ok := parentToChildren[rootID]; ok {
			roots = children
		}
	} else {
		// 找所有根节点
		rootIDs := datasetService.treeUtils.FindRootIDs(toTreeDataNodes(datasetList))
		for _, rid := range rootIDs {
			if children, ok := parentToChildren[rid]; ok {
				roots = append(roots, children...)
			}
		}
	}

	// 递归构建树形选项
	result := make([]vo.Option, 0, len(roots))
	for _, root := range roots {
		result = append(result, datasetService.buildNodeOption(root, parentToChildren)...)
	}

	return result
}

// buildNodeOption 递归构建节点选项
func (datasetService *DatasetService) buildNodeOption(dataset model.SysDataset, parentToChildren map[int64][]model.SysDataset) []vo.Option {
	option := vo.Option{
		Value: dataset.ID,
		Label: dataset.Name,
	}

	// 递归处理子节点
	if children, ok := parentToChildren[dataset.ID]; ok {
		option.Children = make([]vo.Option, 0, len(children))
		for _, child := range children {
			option.Children = append(option.Children, datasetService.buildNodeOption(child, parentToChildren)...)
		}
	}

	return []vo.Option{option}
}

// GetFormData 获取数据集表单数据
func (datasetService *DatasetService) GetFormData(ctx context.Context, id int64) (*bo.DatasetFormBO, error) {
	formRead, err := datasetService.datasetRepo.GetFormData(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询数据集表单失败", err)
	}
	if formRead == nil || formRead.ID == nil {
		return &bo.DatasetFormBO{}, nil
	}

	// 加载统计信息
	stats, statsErr := datasetService.GetDatasetStatistics(id)
	var statsBO *bo.StatisticsBO
	if statsErr != nil {
		logger.Warn("加载数据集统计信息失败", zap.Int64("datasetID", id), zap.Error(statsErr))
		statsBO = &bo.StatisticsBO{
			ItemCount:          0,
			FileCount:          0,
			TotalSize:          0,
			ClearCount:         0,
			HazyCount:          0,
			SceneDistribution:  make(map[string]int64),
			HazeDistribution:   make(map[string]int64),
			FormatDistribution: make(map[string]int64),
		}
	} else {
		statsBO = &bo.StatisticsBO{
			ItemCount:          stats.ItemCount,
			FileCount:          stats.FileCount,
			TotalSize:          stats.TotalSize,
			ClearCount:         stats.ClearCount,
			HazyCount:          stats.HazyCount,
			SceneDistribution:  stats.SceneDistribution,
			HazeDistribution:   stats.HazeDistribution,
			FormatDistribution: stats.FormatDistribution,
		}
	}

	// 转换 ReadModel 为 BO
	return &bo.DatasetFormBO{
		ID:          formRead.ID,
		ParentID:    formRead.ParentID,
		Type:        formRead.Type,
		Name:        formRead.Name,
		Description: formRead.Description,
		Path:        formRead.Path,
		Status:      formRead.Status,
		CreateTime:  formRead.CreateTime.Format("2006-01-02T15:04:05"),
		UpdateTime:  formRead.UpdateTime.Format("2006-01-02T15:04:05"),
		Statistics:  statsBO,
	}, nil
}

// Create 创建数据集
func (datasetService *DatasetService) Create(ctx context.Context, form *bo.DatasetFormBO) error {
	// 父数据集存在性校验
	if form.ParentID != 0 {
		parent, err := datasetService.datasetRepo.FindByID(ctx, form.ParentID)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询父数据集失败", err)
		}
		if parent == nil {
			return common.NewBizError(common.PARAM_ERROR, "父数据集不存在")
		}
	}

	// 同一父数据集下名称唯一性校验
	exists, err := datasetService.datasetRepo.ExistsByParentIDAndName(ctx, form.ParentID, form.Name, 0)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询数据集名称失败", err)
	}
	if exists {
		return common.NewBizError(common.BUSINESS_ERROR, "同一父数据集下已存在同名数据集")
	}

	dataset := &model.SysDataset{
		ParentID:    form.ParentID,
		Type:        form.Type,
		Name:        form.Name,
		Description: form.Description,
		Path:        form.Path,
		Status:      form.Status,
		Deleted:     0,
	}

	if err := datasetService.datasetRepo.Create(ctx, dataset); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "创建数据集失败", err)
	}

	// 失效缓存（使用独立 context 避免请求取消影响缓存操作）
	cacheCtx := context.Background()
	datasetService.invalidateStatsCache(cacheCtx, dataset.ID)
	if form.ParentID != 0 {
		datasetService.invalidateStatsCache(cacheCtx, form.ParentID)
	}
	datasetService.invalidateTreeCache(cacheCtx)

	return nil
}

// Update 更新数据集
func (datasetService *DatasetService) Update(ctx context.Context, id int64, form *bo.DatasetFormBO) error {
	dataset, err := datasetService.datasetRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询数据集失败", err)
	}
	if dataset == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "数据集不存在")
	}

	// 名称变更时校验唯一性
	if form.Name != dataset.Name || form.ParentID != dataset.ParentID {
		exists, err := datasetService.datasetRepo.ExistsByParentIDAndName(ctx, form.ParentID, form.Name, id)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询数据集名称失败", err)
		}
		if exists {
			return common.NewBizError(common.BUSINESS_ERROR, "同一父数据集下已存在同名数据集")
		}
	}

	oldParentID := dataset.ParentID
	dataset.ParentID = form.ParentID
	dataset.Type = form.Type
	dataset.Name = form.Name
	dataset.Description = form.Description
	dataset.Path = form.Path
	dataset.Status = form.Status

	if err := datasetService.datasetRepo.Update(ctx, dataset); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新数据集失败", err)
	}

	cacheCtx := context.Background()
	datasetService.invalidateStatsCache(cacheCtx, id)
	if oldParentID != form.ParentID {
		datasetService.invalidateStatsCache(cacheCtx, oldParentID)
		datasetService.invalidateStatsCache(cacheCtx, form.ParentID)
	}
	datasetService.invalidateTreeCache(cacheCtx)

	return nil
}

// Delete 删除数据集
func (datasetService *DatasetService) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return common.NewBizError(common.PARAM_ERROR, "删除数据为空")
	}

	if err := datasetService.datasetRepo.Delete(ctx, ids); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除数据集失败", err)
	}

	ctx = context.Background()
	for _, id := range ids {
		datasetService.invalidateStatsCache(ctx, id)
	}
	datasetService.invalidateTreeCache(ctx)

	return nil
}

// GetDatasetStatistics 获取数据集统计信息
// 支持缓存，TTL 30分钟
func (datasetService *DatasetService) GetDatasetStatistics(datasetID int64) (*DatasetStatistics, error) {
	ctx := context.Background()
	cacheKey := CACHE_KEY_DATASET_STATS + fmt.Sprintf("%d", datasetID)

	// 1. 尝试从缓存获取
	if datasetService.cache != nil {
		cachedData, err := datasetService.cache.Get(ctx, cacheKey)
		if err == nil && cachedData != "" {
			var stats DatasetStatistics
			if err := json.Unmarshal([]byte(cachedData), &stats); err == nil {
				logger.Debug("统计信息命中缓存", zap.Int64("datasetID", datasetID))
				return &stats, nil
			}
		}
	}

	// 2. 缓存未命中，从数据库计算
	logger.Info("统计信息未命中缓存，开始计算", zap.Int64("datasetID", datasetID))
	stats, err := datasetService.calculateStatisticsFromDB(ctx, datasetID)
	if err != nil {
		return nil, err
	}

	// 3. 写入缓存
	if datasetService.cache != nil {
		if statsJSON, marshalErr := json.Marshal(stats); marshalErr == nil {
			_ = datasetService.cache.Set(ctx, cacheKey, statsJSON, DATASET_STATS_TTL)
		}
	}

	return stats, nil
}

// calculateStatisticsFromDB 从数据库计算统计信息
func (datasetService *DatasetService) calculateStatisticsFromDB(ctx context.Context, datasetID int64) (*DatasetStatistics, error) {
	// 获取叶子节点（使用优化的方法）
	leafIDs, err := datasetService.getLeafDatasetIDsOptimized(ctx, datasetID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "获取叶子节点失败", err)
	}

	if len(leafIDs) == 0 {
		return &DatasetStatistics{
			ItemCount:          0,
			FileCount:          0,
			TotalSize:          0,
			SceneDistribution:  make(map[string]int64),
			HazeDistribution:   make(map[string]int64),
			FormatDistribution: make(map[string]int64),
		}, nil
	}

	// 查询数据项数量
	itemCount, err := datasetService.datasetItemRepo.CountByDatasetIDs(ctx, leafIDs)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "统计数据项数量失败", err)
	}

	// 查询数据项 ID 列表
	itemIDs, err := datasetService.datasetItemRepo.FindIDsByDatasetIDs(ctx, leafIDs)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询数据项ID列表失败", err)
	}

	// 查询项文件
	itemFiles, err := datasetService.itemFileRepo.FindByItemIDs(ctx, itemIDs)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询项文件失败", err)
	}

	fileCount := int64(len(itemFiles))

	stats := &DatasetStatistics{
		ItemCount:          itemCount,
		FileCount:          fileCount,
		TotalSize:          0,
		ClearCount:         0,
		HazyCount:          0,
		SceneDistribution:  make(map[string]int64),
		HazeDistribution:   make(map[string]int64),
		FormatDistribution: make(map[string]int64),
	}

	// 统计清晰图/有雾图数量、雾霾程度分布、场景分布
	fileIDs := make([]int64, 0, len(itemFiles))
	for _, itemFile := range itemFiles {
		fileIDs = append(fileIDs, itemFile.FileID)

		switch itemFile.Type {
		case "clear":
			stats.ClearCount++
		case "hazy":
			stats.HazyCount++
			if itemFile.HazeLevel != nil && *itemFile.HazeLevel != "" {
				stats.HazeDistribution[*itemFile.HazeLevel]++
			}
		}

		if itemFile.SceneType != nil && *itemFile.SceneType != "" {
			stats.SceneDistribution[*itemFile.SceneType]++
		}
	}

	// 查询文件详情
	if len(fileIDs) > 0 {
		files, err := datasetService.fileRepo.FindByIDs(ctx, fileIDs)
		if err == nil {
			for _, file := range files {
				// 统计文件大小
				size := parseSize(file.Size)
				stats.TotalSize += size

				// 统计格式分布
				ext := getExtension(file.Name)
				if ext != "" {
					stats.FormatDistribution[ext]++
				}
			}
		}
	}

	return stats, nil
}

// getLeafDatasetIDsOptimized 优化后的叶子节点计算（一次查询+内存计算）
func (datasetService *DatasetService) getLeafDatasetIDsOptimized(ctx context.Context, datasetID int64) ([]int64, error) {
	cacheKey := "dataset:leaf:" + fmt.Sprintf("%d", datasetID)

	// 1. 尝试从缓存获取
	if datasetService.cache != nil {
		cachedData, err := datasetService.cache.Get(ctx, cacheKey)
		if err == nil && cachedData != "" {
			var leafIDs []int64
			if err := json.Unmarshal([]byte(cachedData), &leafIDs); err == nil {
				return leafIDs, nil
			}
		}
	}

	// 2. 一次性查询所有数据集
	allDatasets, err := datasetService.datasetRepo.FindAll(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询所有数据集失败", err)
	}

	// 3. 使用工具类查找叶子节点
	nodes := toTreeDataNodes(allDatasets)
	leafIDs := datasetService.treeUtils.FindLeafNodesBFS(nodes, datasetID)

	// 4. 写入缓存
	if datasetService.cache != nil {
		if leafIDsJSON, marshalErr := json.Marshal(leafIDs); marshalErr == nil {
			_ = datasetService.cache.Set(ctx, cacheKey, leafIDsJSON, DATASET_LEAF_TTL)
		}
	}

	return leafIDs, nil
}

// invalidateStatsCache 失效统计缓存
func (datasetService *DatasetService) invalidateStatsCache(ctx context.Context, datasetID int64) {
	if datasetService.cache == nil {
		return
	}

	cacheKey := CACHE_KEY_DATASET_STATS + fmt.Sprintf("%d", datasetID)
	if err := datasetService.cache.Delete(ctx, cacheKey); err != nil {
		logger.Warn("失效统计缓存失败", zap.String("key", cacheKey), zap.Error(err))
	}

	// 同时失效叶子节点缓存
	leafCacheKey := "dataset:leaf:" + fmt.Sprintf("%d", datasetID)
	_ = datasetService.cache.Delete(ctx, leafCacheKey)
}

func (datasetService *DatasetService) invalidateTreeCache(ctx context.Context) {
	if datasetService.cache == nil {
		return
	}
	keys := []string{
		CACHE_KEY_DATASET_TREE,
		CACHE_KEY_DATASET_TREE + ":options",
	}
	for _, key := range keys {
		if err := datasetService.cache.Delete(ctx, key); err != nil {
			logger.Warn("失效树缓存失败", zap.String("key", key), zap.Error(err))
		}
	}
}

// ========== 辅助函数 ==========

// toTreeDataNodes 转换为树节点切片
func toTreeDataNodes(datasets []model.SysDataset) []utils.TreeDataNode {
	result := make([]utils.TreeDataNode, 0, len(datasets))
	for i := range datasets {
		result = append(result, &datasets[i])
	}
	return result
}

// parseSize 解析文件大小字符串为字节数
func parseSize(sizeStr string) int64 {
	// 简化实现，假设字符串已经是数字格式
	if sizeStr == "" {
		return 0
	}

	var size int64
	fmt.Sscanf(sizeStr, "%d", &size)
	return size
}

// getExtension 获取文件扩展名
func getExtension(filename string) string {
	idx := strings.LastIndex(filename, ".")
	if idx == -1 {
		return ""
	}
	return strings.ToLower(filename[idx:])
}

// ========== 缓存键常量 ==========

const (
	CACHE_KEY_DATASET_STATS = "dataset:stats:"
	CACHE_KEY_DATASET_TREE  = "dataset:tree"
)

func mapDatasetReadToVO(item read.Dataset) vo.DatasetVO {
	result := vo.DatasetVO{
		ID:          item.ID,
		ParentID:    item.ParentID,
		Type:        item.Type,
		Name:        item.Name,
		Description: item.Description,
		Path:        item.Path,
		Size:        item.Size,
		CreateTime:  item.CreateTime,
		UpdateTime:  item.UpdateTime,
		Status:      item.Status,
	}

	if len(item.Children) > 0 {
		result.Children = make([]vo.DatasetVO, 0, len(item.Children))
		for _, child := range item.Children {
			result.Children = append(result.Children, mapDatasetReadToVO(child))
		}
	}

	return result
}

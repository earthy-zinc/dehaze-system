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
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
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
	datasetRepo repository.IDatasetRepository
	treeUtils   *utils.TreeDataUtils
}

// NewDatasetService 创建数据集服务实例
func NewDatasetService(datasetRepo repository.IDatasetRepository) *DatasetService {
	return &DatasetService{
		datasetRepo: datasetRepo,
		treeUtils:   utils.NewTreeDataUtils(),
	}
}

// getRepo 获取 Repository（兼容零值实例）
func (s *DatasetService) getRepo() repository.IDatasetRepository {
	if s.datasetRepo != nil {
		return s.datasetRepo
	}
	return repository.NewDatasetRepository(global.DB)
}

// SetDatasetRepo 设置 Repository（测试用）
func (s *DatasetService) SetDatasetRepo(repo repository.IDatasetRepository) {
	s.datasetRepo = repo
}

// DatasetStatistics 数据集统计信息
type DatasetStatistics struct {
	ItemCount          int64            `json:"itemCount"`
	FileCount          int64            `json:"fileCount"`
	TotalSize          int64            `json:"totalSize"`
	SceneDistribution  map[string]int64 `json:"sceneDistribution"`
	HazeDistribution   map[string]int64 `json:"hazeDistribution"`
	FormatDistribution map[string]int64 `json:"formatDistribution"`
}

// ====================
// IDatasetService 接口实现
// ====================

// GetPage 数据集分页列表
func (datasetService *DatasetService) GetPage(ctx context.Context, q *query.DatasetQuery) (*vo.PageResult[vo.DatasetVO], error) {
	repo := datasetService.getRepo()
	return repo.FindPage(ctx, q)
}

// GetDatasetOptions 数据集下拉选项
func (datasetService *DatasetService) GetDatasetOptions() (options []vo.Option, err error) {
	ctx := context.Background()
	cacheKey := CACHE_KEY_DATASET_TREE + ":options"

	cachedData, err := global.REDIS.Get(ctx, cacheKey).Result()
	if err == nil && cachedData != "" {
		if err := json.Unmarshal([]byte(cachedData), &options); err == nil {
			logger.Debug("下拉选项命中缓存")
			return options, nil
		}
	}

	var datasetList []model.SysDataset
	err = global.DB.Model(&model.SysDataset{}).
		Where("status = ? AND deleted = ?", 1, 0).
		Select("id, parent_id, name").
		Find(&datasetList).Error

	if err != nil {
		return options, err
	}

	if len(datasetList) == 0 {
		return options, nil
	}

	options = datasetService.buildTreeOptions(datasetList, 0)

	if optionsJSON, marshalErr := json.Marshal(options); marshalErr == nil {
		global.REDIS.Set(ctx, cacheKey, optionsJSON, DATASET_TREE_TTL)
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
	repo := datasetService.getRepo()
	return repo.GetFormData(ctx, id)
}

// Create 创建数据集
func (datasetService *DatasetService) Create(ctx context.Context, form *bo.DatasetFormBO) error {
	repo := datasetService.getRepo()

	dataset := &model.SysDataset{
		ParentID:    form.ParentID,
		Type:        form.Type,
		Name:        form.Name,
		Description: form.Description,
		Path:        form.Path,
		Status:      form.Status,
		Deleted:     0,
	}

	if err := repo.Create(ctx, dataset); err != nil {
		return err
	}

	ctx = context.Background()
	datasetService.invalidateStatsCache(ctx, dataset.ID)
	datasetService.invalidateTreeCache(ctx)

	return nil
}

// Update 更新数据集
func (datasetService *DatasetService) Update(ctx context.Context, id int64, form *bo.DatasetFormBO) error {
	repo := datasetService.getRepo()

	dataset, err := repo.FindByID(ctx, id)
	if err != nil {
		return err
	}
	if dataset == nil {
		return errors.New("数据集不存在")
	}

	dataset.ParentID = form.ParentID
	dataset.Type = form.Type
	dataset.Name = form.Name
	dataset.Description = form.Description
	dataset.Path = form.Path
	dataset.Status = form.Status

	if err := repo.Update(ctx, dataset); err != nil {
		return err
	}

	ctx = context.Background()
	datasetService.invalidateStatsCache(ctx, id)
	datasetService.invalidateTreeCache(ctx)

	return nil
}

// Delete 删除数据集
func (datasetService *DatasetService) Delete(ctx context.Context, ids []int64) error {
	repo := datasetService.getRepo()

	if len(ids) == 0 {
		return errors.New("删除数据为空")
	}

	if err := repo.Delete(ctx, ids); err != nil {
		return err
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
	cachedData, err := global.REDIS.Get(ctx, cacheKey).Result()
	if err == nil && cachedData != "" {
		var stats DatasetStatistics
		if err := json.Unmarshal([]byte(cachedData), &stats); err == nil {
			logger.Debug("统计信息命中缓存", zap.Int64("datasetID", datasetID))
			return &stats, nil
		}
	}

	// 2. 缓存未命中，从数据库计算
	logger.Info("统计信息未命中缓存，开始计算", zap.Int64("datasetID", datasetID))
	stats, err := datasetService.calculateStatisticsFromDB(datasetID)
	if err != nil {
		return nil, err
	}

	// 3. 写入缓存
	if statsJSON, marshalErr := json.Marshal(stats); marshalErr == nil {
		global.REDIS.Set(ctx, cacheKey, statsJSON, DATASET_STATS_TTL)
	}

	return stats, nil
}

// calculateStatisticsFromDB 从数据库计算统计信息
func (datasetService *DatasetService) calculateStatisticsFromDB(datasetID int64) (*DatasetStatistics, error) {
	// 获取叶子节点（使用优化的方法）
	leafIDs, err := datasetService.getLeafDatasetIDsOptimized(datasetID)
	if err != nil {
		return nil, fmt.Errorf("获取叶子节点失败: %w", err)
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

	// 聚合查询统计信息
	var result struct {
		ItemCount int64
		FileCount int64
	}

	// 查询数据项数量
	err = global.DB.Model(&model.SysDatasetItem{}).
		Where("dataset_id IN ?", leafIDs).
		Count(&result.ItemCount).Error
	if err != nil {
		return nil, err
	}

	// 查询文件信息
	var itemFiles []model.SysItemFile
	err = global.DB.Where("item_id IN ?",
		global.DB.Model(&model.SysDatasetItem{}).
			Where("dataset_id IN ?", leafIDs).
			Select("id")).
		Find(&itemFiles).Error
	if err != nil {
		return nil, err
	}

	result.FileCount = int64(len(itemFiles))

	stats := &DatasetStatistics{
		ItemCount:          result.ItemCount,
		FileCount:          result.FileCount,
		TotalSize:          0,
		SceneDistribution:  make(map[string]int64),
		HazeDistribution:   make(map[string]int64),
		FormatDistribution: make(map[string]int64),
	}

	// 统计文件大小、雾霾程度分布、格式分布
	fileIDs := make([]int64, 0, len(itemFiles))
	for _, itemFile := range itemFiles {
		fileIDs = append(fileIDs, itemFile.FileID)

		// 统计雾霾程度分布
		if itemFile.Type == "hazy" && itemFile.Description != nil {
			hazeLevel := extractHazeLevel(*itemFile.Description)
			stats.HazeDistribution[hazeLevel]++
		}
	}

	// 查询文件详情
	if len(fileIDs) > 0 {
		var files []model.SysFile
		err := global.DB.Where("id IN ?", fileIDs).Find(&files).Error
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
func (datasetService *DatasetService) getLeafDatasetIDsOptimized(datasetID int64) ([]int64, error) {
	ctx := context.Background()
	cacheKey := "dataset:leaf:" + fmt.Sprintf("%d", datasetID)

	// 1. 尝试从缓存获取
	cachedData, err := global.REDIS.Get(ctx, cacheKey).Result()
	if err == nil && cachedData != "" {
		var leafIDs []int64
		if err := json.Unmarshal([]byte(cachedData), &leafIDs); err == nil {
			return leafIDs, nil
		}
	}

	// 2. 一次性查询所有数据集
	var allDatasets []model.SysDataset
	err = global.DB.Model(&model.SysDataset{}).
		Where("deleted = ?", 0).
		Find(&allDatasets).Error
	if err != nil {
		return nil, err
	}

	// 3. 使用工具类查找叶子节点
	nodes := toTreeDataNodes(allDatasets)
	leafIDs := datasetService.treeUtils.FindLeafNodesBFS(nodes, datasetID)

	// 4. 写入缓存
	if leafIDsJSON, marshalErr := json.Marshal(leafIDs); marshalErr == nil {
		global.REDIS.Set(ctx, cacheKey, leafIDsJSON, DATASET_LEAF_TTL)
	}

	return leafIDs, nil
}

// invalidateStatsCache 失效统计缓存
func (datasetService *DatasetService) invalidateStatsCache(ctx context.Context, datasetID int64) {
	if global.REDIS == nil {
		return
	}
	cacheKey := CACHE_KEY_DATASET_STATS + fmt.Sprintf("%d", datasetID)
	if err := global.REDIS.Del(ctx, cacheKey).Err(); err != nil {
		logger.Warn("失效统计缓存失败", zap.String("key", cacheKey), zap.Error(err))
	}

	// 同时失效叶子节点缓存
	leafCacheKey := "dataset:leaf:" + fmt.Sprintf("%d", datasetID)
	global.REDIS.Del(ctx, leafCacheKey)
}

// invalidateTreeCache 失效树形缓存
func (datasetService *DatasetService) invalidateTreeCache(ctx context.Context) {
	if global.REDIS == nil {
		return
	}
	keys := []string{
		CACHE_KEY_DATASET_TREE,
		CACHE_KEY_DATASET_TREE + ":options",
	}
	for _, key := range keys {
		if err := global.REDIS.Del(ctx, key).Err(); err != nil {
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

// extractHazeLevel 从描述中提取雾霾程度
func extractHazeLevel(description string) string {
	if description == "" {
		return "unknown"
	}

	switch {
	case contains(description, "light"), contains(description, "轻度"):
		return "light"
	case contains(description, "heavy"), contains(description, "重度"):
		return "heavy"
	case contains(description, "medium"), contains(description, "中度"):
		return "medium"
	default:
		return "medium"
	}
}

// contains 检查字符串包含（不区分大小写）
func contains(s, substr string) bool {
	return len(s) >= len(substr) &&
		(s == substr ||
			len(s) > len(substr) &&
				(s[:len(substr)] == substr ||
					s[len(s)-len(substr):] == substr ||
					containsMiddle(s, substr)))
}

func containsMiddle(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
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
	idx := -1
	for i := len(filename) - 1; i >= 0; i-- {
		if filename[i] == '.' {
			idx = i
			break
		}
	}
	if idx == -1 {
		return ""
	}
	return filename[idx:]
}

// ========== 缓存键常量 ==========

const (
	CACHE_KEY_DATASET_STATS = "dataset:stats:"
	CACHE_KEY_DATASET_TREE  = "dataset:tree"
)

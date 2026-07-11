package dataset

import (
	"context"
	"encoding/json"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
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
	DATASET_STATS_TTL  = 30 * time.Minute
	DATASET_TREE_TTL   = time.Hour
	DATASET_ALL_TTL    = time.Hour
	DATASET_STATSMAP_TTL = 30 * time.Minute
)

type DatasetService struct {
	cache          types.ICache
	datasetRepo    datasetrepo.IDatasetRepository
	datasetItemRepo datasetrepo.IDatasetItemRepository
	statsRepo      datasetrepo.IDatasetStatsRepository
	itemFileRepo   filerepo.IItemFileRepository
	fileRepo       filerepo.IFileRepository
	treeUtils      *utils.TreeDataUtils
}

func NewDatasetService(
	cache types.ICache,
	datasetRepo datasetrepo.IDatasetRepository,
	datasetItemRepo datasetrepo.IDatasetItemRepository,
	statsRepo datasetrepo.IDatasetStatsRepository,
	itemFileRepo filerepo.IItemFileRepository,
	fileRepo filerepo.IFileRepository,
) *DatasetService {
	if datasetRepo == nil {
		panic("DatasetService: datasetRepo 未初始化")
	}
	if datasetItemRepo == nil {
		panic("DatasetService: datasetItemRepo 未初始化")
	}
	if statsRepo == nil {
		panic("DatasetService: statsRepo 未初始化")
	}
	if itemFileRepo == nil {
		panic("DatasetService: itemFileRepo 未初始化")
	}
	if fileRepo == nil {
		panic("DatasetService: fileRepo 未初始化")
	}

	return &DatasetService{
		cache:          cache,
		datasetRepo:    datasetRepo,
		datasetItemRepo: datasetItemRepo,
		statsRepo:      statsRepo,
		itemFileRepo:   itemFileRepo,
		fileRepo:       fileRepo,
		treeUtils:      utils.NewTreeDataUtils(),
	}
}

func (s *DatasetService) SetDatasetRepo(repo datasetrepo.IDatasetRepository) {
	s.datasetRepo = repo
}

func (datasetService *DatasetService) getAllDatasets(ctx context.Context) ([]model.SysDataset, error) {
	cacheKey := "dataset:all"

	if datasetService.cache != nil {
		cachedData, err := datasetService.cache.Get(ctx, cacheKey)
		if err == nil && cachedData != "" {
			var datasets []model.SysDataset
			if err := json.Unmarshal([]byte(cachedData), &datasets); err == nil {
				return datasets, nil
			}
		}
	}

	datasets, err := datasetService.datasetRepo.FindAll(ctx)
	if err != nil {
		return nil, err
	}

	if datasetService.cache != nil && len(datasets) > 0 {
		if dataJSON, marshalErr := json.Marshal(datasets); marshalErr == nil {
			_ = datasetService.cache.Set(ctx, cacheKey, dataJSON, DATASET_ALL_TTL)
		}
	}

	return datasets, nil
}

func (datasetService *DatasetService) getAllDatasetStats(ctx context.Context) (map[int64]*vo.DatasetStatistics, error) {
	cacheKey := "dataset:statsMap:all"

	if datasetService.cache != nil {
		cachedData, err := datasetService.cache.Get(ctx, cacheKey)
		if err == nil && cachedData != "" {
			var statsMap map[int64]*vo.DatasetStatistics
			if err := json.Unmarshal([]byte(cachedData), &statsMap); err == nil {
				logger.Debug("所有数据集统计信息命中缓存")
				return statsMap, nil
			}
		}
	}

	startTime := time.Now()
	logger.Debug("开始计算所有数据集统计信息...")

	allDatasets, err := datasetService.getAllDatasets(ctx)
	if err != nil {
		return nil, err
	}

	statsMap := make(map[int64]*vo.DatasetStatistics)
	for _, ds := range allDatasets {
		statsMap[ds.ID] = createEmptyStats()
	}

	if len(allDatasets) == 0 {
		return statsMap, nil
	}

	childParentIDs := make(map[int64]bool)
	for _, d := range allDatasets {
		if d.ParentID != 0 {
			childParentIDs[d.ParentID] = true
		}
	}

	var leafIDs []int64
	for _, d := range allDatasets {
		if !childParentIDs[d.ID] {
			leafIDs = append(leafIDs, d.ID)
		}
	}

	if len(leafIDs) > 0 {
		logger.Debug("发现叶子数据集", zap.Int("count", len(leafIDs)))

		itemResults, err := datasetService.datasetItemRepo.CountItemsPerDataset(ctx, leafIDs)
		if err == nil {
			for _, r := range itemResults {
				if stats, ok := statsMap[r.DatasetID]; ok {
					stats.ItemCount = r.Cnt
				}
			}
		}

		statsResults, err := datasetService.statsRepo.CountDatasetStatsBatch(ctx, leafIDs)
		if err == nil {
			for _, r := range statsResults {
				if stats, ok := statsMap[r.DatasetID]; ok {
					stats.FileCount = r.FileCount
					stats.TotalSize = r.TotalSize
					stats.ClearCount = r.ClearCount
					stats.HazyCount = r.HazyCount
				}
			}
		}

		sceneResults, err := datasetService.statsRepo.CountSceneDistributionBatch(ctx, leafIDs)
		if err == nil {
			for _, r := range sceneResults {
				if stats, ok := statsMap[r.DatasetID]; ok {
					stats.SceneDistribution[r.Key] += r.Cnt
				}
			}
		}

		hazeResults, err := datasetService.statsRepo.CountHazeDistributionBatch(ctx, leafIDs)
		if err == nil {
			for _, r := range hazeResults {
				if stats, ok := statsMap[r.DatasetID]; ok {
					stats.HazeDistribution[r.Key] += r.Cnt
				}
			}
		}

		formatResults, err := datasetService.statsRepo.CountFormatDistributionBatch(ctx, leafIDs)
		if err == nil {
			for _, r := range formatResults {
				if stats, ok := statsMap[r.DatasetID]; ok {
					stats.FormatDistribution[r.Key] += r.Cnt
				}
			}
		}
	}

	parentToChildren := make(map[int64][]int64)
	for _, ds := range allDatasets {
		if ds.ParentID != 0 {
			parentToChildren[ds.ParentID] = append(parentToChildren[ds.ParentID], ds.ID)
		}
	}

	processed := make(map[int64]bool)
	queue := make([]int64, 0, len(leafIDs))
	for _, id := range leafIDs {
		queue = append(queue, id)
		processed[id] = true
	}

	for len(queue) > 0 {
		currentID := queue[0]
		queue = queue[1:]

		var current *model.SysDataset
		for i := range allDatasets {
			if allDatasets[i].ID == currentID {
				current = &allDatasets[i]
				break
			}
		}
		if current == nil || current.ParentID == 0 {
			continue
		}

		parentID := current.ParentID
		parentStats := statsMap[parentID]
		childStats := statsMap[currentID]
		if parentStats != nil && childStats != nil {
			mergeStats(parentStats, childStats)
		}

		siblings := parentToChildren[parentID]
		allSiblingsProcessed := true
		for _, sid := range siblings {
			if !processed[sid] {
				allSiblingsProcessed = false
				break
			}
		}
		if allSiblingsProcessed && !processed[parentID] {
			processed[parentID] = true
			queue = append(queue, parentID)
		}
	}

	costMs := time.Since(startTime).Milliseconds()
	logger.Info("所有数据集统计信息计算完成", zap.Int64("costMs", costMs), zap.Int("leafCount", len(leafIDs)))

	if datasetService.cache != nil {
		if dataJSON, marshalErr := json.Marshal(statsMap); marshalErr == nil {
			_ = datasetService.cache.Set(ctx, cacheKey, dataJSON, DATASET_STATSMAP_TTL)
		}
	}

	return statsMap, nil
}

func createEmptyStats() *vo.DatasetStatistics {
	return &vo.DatasetStatistics{
		ItemCount:          0,
		FileCount:          0,
		TotalSize:          0,
		ClearCount:         0,
		HazyCount:          0,
		SceneDistribution:  make(map[string]int64),
		HazeDistribution:   make(map[string]int64),
		FormatDistribution: make(map[string]int64),
	}
}

func mergeStats(parent, child *vo.DatasetStatistics) {
	parent.ItemCount += child.ItemCount
	parent.FileCount += child.FileCount
	parent.TotalSize += child.TotalSize
	parent.ClearCount += child.ClearCount
	parent.HazyCount += child.HazyCount
	for k, v := range child.SceneDistribution {
		parent.SceneDistribution[k] += v
	}
	for k, v := range child.HazeDistribution {
		parent.HazeDistribution[k] += v
	}
	for k, v := range child.FormatDistribution {
		parent.FormatDistribution[k] += v
	}
}

func (datasetService *DatasetService) evictAllDatasetsCache(ctx context.Context) {
	if datasetService.cache == nil {
		return
	}
	keys := []string{
		"dataset:all",
		"dataset:statsMap:all",
		"dataset:tree",
		"dataset:tree:options",
	}
	for _, key := range keys {
		if err := datasetService.cache.Delete(ctx, key); err != nil {
			logger.Warn("失效缓存失败", zap.String("key", key), zap.Error(err))
		}
	}
}

func (datasetService *DatasetService) GetPage(ctx context.Context, q *query.DatasetQuery) (*vo.PageResult[vo.DatasetVO], error) {
	rootDatasets, total, err := datasetService.datasetRepo.FindRootPage(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询数据集分页列表失败", err)
	}
	if len(rootDatasets) == 0 {
		return &vo.PageResult[vo.DatasetVO]{List: []vo.DatasetVO{}, Total: total}, nil
	}

	rootIDs := make([]int64, 0, len(rootDatasets))
	for _, d := range rootDatasets {
		rootIDs = append(rootIDs, d.ID)
	}

	allDirectChildren, err := datasetService.datasetRepo.FindByParentIDs(ctx, rootIDs)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询子数据集失败", err)
	}

	directChildrenMap := make(map[int64][]model.SysDataset)
	childIDs := make([]int64, 0)
	for _, c := range allDirectChildren {
		directChildrenMap[c.ParentID] = append(directChildrenMap[c.ParentID], c)
		childIDs = append(childIDs, c.ID)
	}

	allParentIDs := append(rootIDs, childIDs...)
	hasChildrenMap, err := datasetService.datasetRepo.CountHasChildren(ctx, allParentIDs)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询子节点标记失败", err)
	}

	statsMap, err := datasetService.getAllDatasetStats(ctx)
	if err != nil {
		logger.Warn("获取统计信息失败", zap.Error(err))
		statsMap = make(map[int64]*vo.DatasetStatistics)
	}

	voList := make([]vo.DatasetVO, 0, len(rootDatasets))
	for _, root := range rootDatasets {
		stats := statsMap[root.ID]
		rootVO := datasetService.entityToVO(&root, stats, hasChildrenMap[root.ID])

		directChildren := directChildrenMap[root.ID]
		childVOs := make([]vo.DatasetVO, 0, len(directChildren))
		for _, child := range directChildren {
			childStats := statsMap[child.ID]
			childVOs = append(childVOs, datasetService.entityToVO(&child, childStats, hasChildrenMap[child.ID]))
		}
		rootVO.Children = childVOs
		voList = append(voList, rootVO)
	}

	return &vo.PageResult[vo.DatasetVO]{
		List:  voList,
		Total: total,
	}, nil
}

func (datasetService *DatasetService) GetChildren(ctx context.Context, parentID int64) ([]vo.DatasetVO, error) {
	if parentID <= 0 {
		return []vo.DatasetVO{}, nil
	}

	children, err := datasetService.datasetRepo.FindByParentID(ctx, parentID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询子数据集失败", err)
	}
	if len(children) == 0 {
		return []vo.DatasetVO{}, nil
	}

	childIDs := make([]int64, 0, len(children))
	for _, c := range children {
		childIDs = append(childIDs, c.ID)
	}

	hasChildrenMap, err := datasetService.datasetRepo.CountHasChildren(ctx, childIDs)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询子节点标记失败", err)
	}

	statsMap, err := datasetService.getAllDatasetStats(ctx)
	if err != nil {
		logger.Warn("获取统计信息失败", zap.Error(err))
		statsMap = make(map[int64]*vo.DatasetStatistics)
	}

	result := make([]vo.DatasetVO, 0, len(children))
	for _, child := range children {
		stats := statsMap[child.ID]
		childVO := datasetService.entityToVO(&child, stats, hasChildrenMap[child.ID])
		childVO.Children = []vo.DatasetVO{}
		result = append(result, childVO)
	}

	return result, nil
}

// GetTree 获取完整数据集树（查询所有数据集，BFS 内存构建树）
func (datasetService *DatasetService) GetTree(ctx context.Context) ([]vo.DatasetVO, error) {
	allDatasets, err := datasetService.getAllDatasets(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询数据集列表失败", err)
	}
	if len(allDatasets) == 0 {
		return []vo.DatasetVO{}, nil
	}

	// 收集所有 parentID 用于判断是否有子节点
	allIDs := make(map[int64]bool)
	childParentIDs := make(map[int64]bool)
	for _, d := range allDatasets {
		allIDs[d.ID] = true
	}
	for _, d := range allDatasets {
		if d.ParentID > 0 && allIDs[d.ParentID] {
			childParentIDs[d.ParentID] = true
		}
	}

	statsMap, err := datasetService.getAllDatasetStats(ctx)
	if err != nil {
		logger.Warn("获取统计信息失败", zap.Error(err))
		statsMap = make(map[int64]*vo.DatasetStatistics)
	}

	// 构建 parentID → children 映射
	childrenMap := make(map[int64][]model.SysDataset)
	for _, d := range allDatasets {
		childrenMap[d.ParentID] = append(childrenMap[d.ParentID], d)
	}

	// 构建 VO 映射
	voMap := make(map[int64]vo.DatasetVO, len(allDatasets))
	for _, d := range allDatasets {
		stats := statsMap[d.ID]
		voMap[d.ID] = datasetService.entityToVO(&d, stats, childParentIDs[d.ID])
	}

	// BFS 构建树
	tree := make([]vo.DatasetVO, 0)
	queue := make([]int64, 0, len(childrenMap[0]))
	for _, root := range childrenMap[0] {
		queue = append(queue, root.ID)
		tree = append(tree, voMap[root.ID])
	}

	for len(queue) > 0 {
		currentID := queue[0]
		queue = queue[1:]

		children := childrenMap[currentID]
		if len(children) == 0 {
			continue
		}

		parentVO := voMap[currentID]
		childVOs := make([]vo.DatasetVO, 0, len(children))
		for _, child := range children {
			childVOs = append(childVOs, voMap[child.ID])
			queue = append(queue, child.ID)
		}
		parentVO.Children = childVOs
		voMap[currentID] = parentVO
	}

	// 重建引用（因为 children 按值复制）
	for i := range tree {
		tree[i] = rebuildVOChildren(&tree[i], voMap)
	}

	return tree, nil
}

func rebuildVOChildren(vo *vo.DatasetVO, voMap map[int64]vo.DatasetVO) vo.DatasetVO {
	result := voMap[vo.ID]
	for i := range result.Children {
		result.Children[i] = voMap[result.Children[i].ID]
	}
	return result
}

func (datasetService *DatasetService) entityToVO(entity *model.SysDataset, stats *vo.DatasetStatistics, hasChildren bool) vo.DatasetVO {
	voItem := vo.DatasetVO{
		ID:          entity.ID,
		ParentID:    entity.ParentID,
		Type:        entity.Type,
		Name:        entity.Name,
		Description: entity.Description,
		Path:        entity.Path,
		Size:        entity.Size,
		HasChildren: hasChildren,
		Children:    []vo.DatasetVO{},
		Status:      int(entity.Status),
		Statistics:  stats,
		CreateTime:  entity.CreatedAt,
		UpdateTime:  entity.UpdatedAt,
	}
	if stats != nil {
		voItem.Total = stats.FileCount
	}
	return voItem
}

func contains(ids []int64, id int64) bool {
	for _, v := range ids {
		if v == id {
			return true
		}
	}
	return false
}

func (datasetService *DatasetService) GetDatasetOptions() (options []vo.Option, err error) {
	ctx := context.Background()
	cacheKey := "dataset:tree:options"

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

func (datasetService *DatasetService) buildTreeOptions(datasetList []model.SysDataset, rootID int64) []vo.Option {
	if len(datasetList) == 0 {
		return []vo.Option{}
	}

	parentToChildren := make(map[int64][]model.SysDataset)
	for _, dataset := range datasetList {
		parentID := dataset.ParentID
		parentToChildren[parentID] = append(parentToChildren[parentID], dataset)
	}

	var roots []model.SysDataset
	if rootID != 0 {
		if children, ok := parentToChildren[rootID]; ok {
			roots = children
		}
	} else {
		rootIDs := datasetService.treeUtils.FindRootIDs(toTreeDataNodes(datasetList))
		for _, rid := range rootIDs {
			if children, ok := parentToChildren[rid]; ok {
				roots = append(roots, children...)
			}
		}
	}

	result := make([]vo.Option, 0, len(roots))
	for _, root := range roots {
		result = append(result, datasetService.buildNodeOption(root, parentToChildren)...)
	}

	return result
}

func (datasetService *DatasetService) buildNodeOption(dataset model.SysDataset, parentToChildren map[int64][]model.SysDataset) []vo.Option {
	option := vo.Option{
		Value: dataset.ID,
		Label: dataset.Name,
	}

	if children, ok := parentToChildren[dataset.ID]; ok {
		option.Children = make([]vo.Option, 0, len(children))
		for _, child := range children {
			option.Children = append(option.Children, datasetService.buildNodeOption(child, parentToChildren)...)
		}
	}

	return []vo.Option{option}
}

func (datasetService *DatasetService) GetFormData(ctx context.Context, id int64) (*bo.DatasetFormBO, error) {
	dataset, err := datasetService.datasetRepo.GetFormData(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询数据集表单失败", err)
	}
	if dataset == nil {
		return &bo.DatasetFormBO{}, nil
	}

	statsMap, err := datasetService.getAllDatasetStats(ctx)
	var statsBO *bo.StatisticsBO
	if err != nil || statsMap[id] == nil {
		empty := createEmptyStats()
		statsBO = &bo.StatisticsBO{
			ItemCount:          empty.ItemCount,
			FileCount:          empty.FileCount,
			TotalSize:          empty.TotalSize,
			ClearCount:         empty.ClearCount,
			HazyCount:          empty.HazyCount,
			SceneDistribution:  empty.SceneDistribution,
			HazeDistribution:   empty.HazeDistribution,
			FormatDistribution: empty.FormatDistribution,
		}
	} else {
		stats := statsMap[id]
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

	idPtr := dataset.ID
	return &bo.DatasetFormBO{
		ID:          &idPtr,
		ParentID:    dataset.ParentID,
		Type:        dataset.Type,
		Name:        dataset.Name,
		Description: dataset.Description,
		Path:        dataset.Path,
		Status:      dataset.Status,
		CreateTime:  dataset.CreatedAt.Format("2006-01-02T15:04:05"),
		UpdateTime:  dataset.UpdatedAt.Format("2006-01-02T15:04:05"),
		Statistics:  statsBO,
	}, nil
}

func (datasetService *DatasetService) Create(ctx context.Context, form *bo.DatasetFormBO) error {
	if form.ParentID != 0 {
		parent, err := datasetService.datasetRepo.FindByID(ctx, form.ParentID)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询父数据集失败", err)
		}
		if parent == nil {
			return common.NewBizError(common.PARAM_ERROR, "父数据集不存在")
		}
	}

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

	cacheCtx := context.Background()
	datasetService.evictAllDatasetsCache(cacheCtx)

	return nil
}

func (datasetService *DatasetService) Update(ctx context.Context, id int64, form *bo.DatasetFormBO) error {
	dataset, err := datasetService.datasetRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询数据集失败", err)
	}
	if dataset == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "数据集不存在")
	}

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
	datasetService.evictAllDatasetsCache(cacheCtx)
	_ = oldParentID

	return nil
}

func (datasetService *DatasetService) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return common.NewBizError(common.PARAM_ERROR, "删除数据为空")
	}

	if err := datasetService.datasetRepo.Delete(ctx, ids); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除数据集失败", err)
	}

	cacheCtx := context.Background()
	datasetService.evictAllDatasetsCache(cacheCtx)

	return nil
}

func toTreeDataNodes(datasets []model.SysDataset) []utils.TreeDataNode {
	result := make([]utils.TreeDataNode, 0, len(datasets))
	for i := range datasets {
		result = append(result, &datasets[i])
	}
	return result
}

const (
	CACHE_KEY_DATASET_STATS = "dataset:stats:"
	CACHE_KEY_DATASET_TREE  = "dataset:tree"
)

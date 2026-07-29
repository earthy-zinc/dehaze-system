package algorithm

import (
	"context"
	"fmt"
	"math"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/service/mapper"
	algorepo "github.com/earthyzinc/dehaze-go/internal/repository/algorithm"
	predlog "github.com/earthyzinc/dehaze-go/internal/repository/pred_log"
)

// AlgorithmService 算法服务
type AlgorithmService struct {
	algorithmRepo algorepo.IAlgorithmRepository
	predLogRepo   predlog.IPredLogRepository
	treeUtils     *utils.TreeDataUtils
}

// NewAlgorithmService 创建算法服务实例
func NewAlgorithmService(algorithmRepo algorepo.IAlgorithmRepository, predLogRepo predlog.IPredLogRepository) *AlgorithmService {
	return &AlgorithmService{
		algorithmRepo: algorithmRepo,
		predLogRepo:   predLogRepo,
		treeUtils:     utils.NewTreeDataUtils(),
	}
}

// ====================
// IAlgorithmService 接口实现
// ====================

// GetPage 算法分页列表
func (s *AlgorithmService) GetPage(ctx context.Context, q *query.AlgorithmQuery) (*vo.PageResult[vo.AlgorithmVO], error) {
	readResult, err := s.algorithmRepo.FindPage(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法分页列表失败", err)
	}
	if readResult == nil {
		return &vo.PageResult[vo.AlgorithmVO]{List: []vo.AlgorithmVO{}, Total: 0}, nil
	}

	voList := make([]vo.AlgorithmVO, 0, len(readResult.List))
	for _, item := range readResult.List {
		voList = append(voList, vo.AlgorithmVO{
			ID:          item.ID,
			ParentID:    item.ParentID,
			Name:        item.Name,
			Type:        item.Type,
			Img:         item.Img,
			Description: item.Description,
			Path:        item.Path,
			Flops:       item.Flops,
			Params:      item.Params,
			ImportPath:  item.ImportPath,
			Status:      item.Status,
			Size:        item.Size,
			Children:    mapAlgorithmReadChildren(item.Children),
		})
	}

	return &vo.PageResult[vo.AlgorithmVO]{
		List:  voList,
		Total: readResult.Total,
	}, nil
}

// GetTree 获取算法树形列表（对齐 Java 树形表格格式）
func (s *AlgorithmService) GetTree(ctx context.Context, q *query.AlgorithmQuery) ([]vo.AlgorithmVO, error) {
	algorithms, err := s.algorithmRepo.FindAll(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法列表失败", err)
	}

	// 按 ParentID 分组，O(n) 构建树形结构
	childrenMap := make(map[int64][]read.Algorithm)
	for _, algo := range algorithms {
		childrenMap[algo.ParentID] = append(childrenMap[algo.ParentID], algo)
	}

	tree := make([]vo.AlgorithmVO, 0)
	for _, algo := range algorithms {
		if algo.ParentID == 0 {
			tree = append(tree, buildAlgorithmVO(algo, childrenMap))
		}
	}
	return tree, nil
}

// buildAlgorithmVO 递归构建算法 VO（使用 map 索引，O(n) 复杂度）
func buildAlgorithmVO(algo read.Algorithm, childrenMap map[int64][]read.Algorithm) vo.AlgorithmVO {
	voItem := vo.AlgorithmVO{
		ID:          algo.ID,
		ParentID:    algo.ParentID,
		Name:        algo.Name,
		Type:        algo.Type,
		Img:         algo.Img,
		Description: algo.Description,
		Path:        algo.Path,
		Flops:       algo.Flops,
		Params:      algo.Params,
		ImportPath:  algo.ImportPath,
		Status:      algo.Status,
		Size:        algo.Size,
	}
	for _, child := range childrenMap[algo.ID] {
		voItem.Children = append(voItem.Children, buildAlgorithmVO(child, childrenMap))
	}
	return voItem
}

// GetOptions 获取算法下拉选项
func (s *AlgorithmService) GetOptions(ctx context.Context) ([]vo.Option, error) {
	readOptions, err := s.algorithmRepo.FindOptions(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法选项失败", err)
	}

	return mapper.OptionsFromRead(readOptions), nil
}

// ListAll 获取所有算法扁平列表（不构建树形）
func (s *AlgorithmService) ListAll(ctx context.Context) ([]vo.AlgorithmVO, error) {
	algorithms, err := s.algorithmRepo.FindAll(ctx, &query.AlgorithmQuery{})
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法列表失败", err)
	}
	voList := make([]vo.AlgorithmVO, 0, len(algorithms))
	for _, item := range algorithms {
		voList = append(voList, vo.AlgorithmVO{
			ID:          item.ID,
			ParentID:    item.ParentID,
			Name:        item.Name,
			Type:        item.Type,
			Img:         item.Img,
			Description: item.Description,
			Path:        item.Path,
			Flops:       item.Flops,
			Params:      item.Params,
			ImportPath:  item.ImportPath,
			Status:      item.Status,
			Size:        item.Size,
		})
	}
	return voList, nil
}

// GetFormData 获取算法表单数据
func (s *AlgorithmService) GetFormData(ctx context.Context, id int64) (*bo.AlgorithmFormBO, error) {
	algorithm, err := s.algorithmRepo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法失败", err)
	}
	if algorithm == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "算法不存在")
	}

	form := &bo.AlgorithmFormBO{
		ID:          algorithm.ID,
		ParentID:    algorithm.ParentID,
		Type:        algorithm.Type,
		Name:        algorithm.Name,
		Path:        algorithm.Path,
		ImportPath:  algorithm.ImportPath,
		Description: algorithm.Description,
		Status:      algorithm.Status,
	}

	return form, nil
}

// Create 创建算法
func (s *AlgorithmService) Create(ctx context.Context, form *bo.AlgorithmFormBO) (int64, error) {
	// 如果父节点ID不为0，检查父节点是否存在
	if form.ParentID != 0 {
		parentAlgorithm, err := s.algorithmRepo.FindByID(ctx, form.ParentID)
		if err != nil {
			return 0, common.WrapBizError(common.DATABASE_ERROR, "查询父算法失败", err)
		}
		if parentAlgorithm == nil {
			return 0, common.NewBizError(common.RESOURCE_NOT_FOUND, "父算法不存在")
		}
	}

	algorithm := &model.SysAlgorithm{
		ParentID:    form.ParentID,
		Type:        form.Type,
		Name:        form.Name,
		Path:        form.Path,
		ImportPath:  form.ImportPath,
		Description: form.Description,
		Status:      int8(form.Status),
	}

	if err := s.algorithmRepo.Create(ctx, algorithm); err != nil {
		return 0, common.WrapBizError(common.DATABASE_ERROR, "创建算法失败", err)
	}
	return algorithm.ID, nil
}

// Update 更新算法
func (s *AlgorithmService) Update(ctx context.Context, id int64, form *bo.AlgorithmFormBO) error {
	// 校验算法是否存在
	oldAlgorithm, err := s.algorithmRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询算法失败", err)
	}
	if oldAlgorithm == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "算法不存在")
	}

	// 如果父节点ID不为0，检查父节点是否存在
	if form.ParentID != 0 && form.ParentID != id {
		parentAlgorithm, err := s.algorithmRepo.FindByID(ctx, form.ParentID)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询父算法失败", err)
		}
		if parentAlgorithm == nil {
			return common.NewBizError(common.RESOURCE_NOT_FOUND, "父算法不存在")
		}
	}

	// 更新算法信息
	oldAlgorithm.ParentID = form.ParentID
	oldAlgorithm.Type = form.Type
	oldAlgorithm.Name = form.Name
	oldAlgorithm.Path = form.Path
	oldAlgorithm.ImportPath = form.ImportPath
	oldAlgorithm.Description = form.Description
	oldAlgorithm.Status = int8(form.Status)

	if err := s.algorithmRepo.Update(ctx, oldAlgorithm); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新算法失败", err)
	}
	return nil
}

// Delete 删除算法（级联删除子孙算法）
func (s *AlgorithmService) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return common.NewBizError(common.PARAM_ERROR, "请选择要删除的算法")
	}

	allAlgorithms, err := s.algorithmRepo.FindAll(ctx, &query.AlgorithmQuery{})
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询算法失败", err)
	}

	existingIDs := make(map[int64]bool, len(allAlgorithms))
	for i := range allAlgorithms {
		existingIDs[allAlgorithms[i].ID] = true
	}
	for _, id := range ids {
		if !existingIDs[id] {
			return common.NewBizError(common.RESOURCE_NOT_FOUND, "算法不存在")
		}
	}

	nodes := make([]utils.TreeDataNode, 0, len(allAlgorithms))
	for i := range allAlgorithms {
		nodes = append(nodes, &allAlgorithms[i])
	}

	allDeleteIDs := make(map[int64]bool)
	for _, id := range ids {
		allDeleteIDs[id] = true
		for _, childID := range s.treeUtils.GetDescendantIDs(nodes, id) {
			allDeleteIDs[childID] = true
		}
	}

	idsToDelete := make([]int64, 0, len(allDeleteIDs))
	for id := range allDeleteIDs {
		idsToDelete = append(idsToDelete, id)
	}

	if err := s.algorithmRepo.Delete(ctx, idsToDelete); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除算法失败", err)
	}
	return nil
}

// UpdateStatus 更新算法状态（含状态流转校验）
func (s *AlgorithmService) UpdateStatus(ctx context.Context, id int64, status int8) error {
	// 1. 查询当前算法
	algorithm, err := s.algorithmRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询算法失败", err)
	}
	if algorithm == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "算法不存在")
	}

	// 2. 校验状态流转合法性
	if !bo.CanTransitionTo(algorithm.Status, status) {
		return common.NewBizError(common.DATA_STATE_NOT_ALLOW,
			fmt.Sprintf("不允许将算法状态从 %d 变更为 %d", algorithm.Status, status))
	}

	// 3. 执行状态更新
	if err := s.algorithmRepo.UpdateStatus(ctx, id, status); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新算法状态失败", err)
	}
	return nil
}

// Compare 批量查询算法用于对比
func (s *AlgorithmService) Compare(ctx context.Context, ids []int64) ([]model.SysAlgorithm, error) {
	if len(ids) == 0 {
		return nil, common.NewBizError(common.PARAM_ERROR, "算法ID列表不能为空")
	}
	algorithms := make([]model.SysAlgorithm, 0, len(ids))
	for _, id := range ids {
		a, err := s.algorithmRepo.FindByID(ctx, id)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法失败", err)
		}
		if a == nil {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, fmt.Sprintf("算法ID %d 不存在", id))
		}
		algorithms = append(algorithms, *a)
	}
	return algorithms, nil
}

// GetVersionHistory 获取算法版本历史
// 查询 sys_algorithm_version 表，按 create_time 降序排序
func (s *AlgorithmService) GetVersionHistory(ctx context.Context, algorithmID int64) ([]vo.AlgorithmVersionVO, error) {
	versions, err := s.algorithmRepo.FindVersionsByAlgorithmID(ctx, algorithmID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法版本历史失败", err)
	}

	result := make([]vo.AlgorithmVersionVO, 0, len(versions))
	for _, v := range versions {
		var isActive *bool
		if v.IsActive != nil {
			b := *v.IsActive != 0
			isActive = &b
		}
		result = append(result, vo.AlgorithmVersionVO{
			ID:          v.ID,
			AlgorithmID: v.AlgorithmID,
			Version:     v.Version,
			ChangeLog:   v.ChangeLog,
			Status:      v.Status,
			IsActive:    isActive,
			ModelFileID: v.ModelFileID,
			CreateTime:  v.CreatedAt,
		})
	}
	return result, nil
}

// GetMonitorData 获取算法监控数据
// 查询 sys_pred_log 表统计
func (s *AlgorithmService) GetMonitorData(ctx context.Context, algorithmID int64) (*vo.AlgorithmMonitorVO, error) {
	stats, err := s.predLogRepo.GetMonitorStats(ctx, algorithmID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询算法监控数据失败", err)
	}

	monitor := &vo.AlgorithmMonitorVO{
		CallCount:      stats.CallCount,
		TodayCallCount: stats.TodayCallCount,
		AvgTime:        math.Round(stats.AvgTime*100) / 100,
	}
	if stats.CallCount > 0 {
		rate := float64(stats.SuccessCount) / float64(stats.CallCount) * 100
		monitor.SuccessRate = math.Round(rate*100) / 100
	}
	return monitor, nil
}

func mapAlgorithmReadChildren(children []read.Algorithm) []vo.AlgorithmVO {
	if len(children) == 0 {
		return []vo.AlgorithmVO{}
	}

	result := make([]vo.AlgorithmVO, 0, len(children))
	for _, child := range children {
		result = append(result, vo.AlgorithmVO{
			ID:          child.ID,
			ParentID:    child.ParentID,
			Name:        child.Name,
			Type:        child.Type,
			Img:         child.Img,
			Description: child.Description,
			Path:        child.Path,
			Flops:       child.Flops,
			Params:      child.Params,
			ImportPath:  child.ImportPath,
			Status:      child.Status,
			Size:        child.Size,
			Children:    mapAlgorithmReadChildren(child.Children),
		})
	}

	return result
}


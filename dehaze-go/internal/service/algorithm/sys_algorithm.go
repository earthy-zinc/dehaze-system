package algorithm

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/service/mapper"
	algorepo "github.com/earthyzinc/dehaze-go/internal/repository/algorithm"
)

// AlgorithmService 算法服务
type AlgorithmService struct {
	algorithmRepo algorepo.IAlgorithmRepository
}

// NewAlgorithmService 创建算法服务实例
func NewAlgorithmService(algorithmRepo algorepo.IAlgorithmRepository) *AlgorithmService {
	return &AlgorithmService{algorithmRepo: algorithmRepo}
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

	// 构建树形结构（parent_id == 0 为根节点）
	tree := make([]vo.AlgorithmVO, 0)
	for _, algo := range algorithms {
		if algo.ParentID == 0 {
			tree = append(tree, mapAlgorithmToVO(algo, algorithms))
		}
	}
	return tree, nil
}

func mapAlgorithmToVO(algo read.Algorithm, all []read.Algorithm) vo.AlgorithmVO {
	voItem := vo.AlgorithmVO{
		ID:          algo.ID,
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
	for _, child := range all {
		if child.ParentID == algo.ID {
			voItem.Children = append(voItem.Children, mapAlgorithmToVO(child, all))
		}
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
		Status:      int(algorithm.Status),
	}

	return form, nil
}

// Create 创建算法
func (s *AlgorithmService) Create(ctx context.Context, form *bo.AlgorithmFormBO) error {
	// 如果父节点ID不为0，检查父节点是否存在
	if form.ParentID != 0 {
		parentAlgorithm, err := s.algorithmRepo.FindByID(ctx, form.ParentID)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询父算法失败", err)
		}
		if parentAlgorithm == nil {
			return common.NewBizError(common.RESOURCE_NOT_FOUND, "父算法不存在")
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
		return common.WrapBizError(common.DATABASE_ERROR, "创建算法失败", err)
	}
	return nil
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

// Delete 删除算法
func (s *AlgorithmService) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return common.NewBizError(common.PARAM_ERROR, "请选择要删除的算法")
	}

	// 检查是否有子算法
	hasChildren, err := s.algorithmRepo.HasChildrenByParentIDs(ctx, ids)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "检查子算法失败", err)
	}

	if hasChildren {
		return common.NewBizError(common.BUSINESS_ERROR, "存在子算法，无法删除")
	}

	if err := s.algorithmRepo.Delete(ctx, ids); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除算法失败", err)
	}
	return nil
}

// UpdateStatus 更新算法状态
func (s *AlgorithmService) UpdateStatus(ctx context.Context, id int64, status int8) error {
	if err := s.algorithmRepo.UpdateStatus(ctx, id, status); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新算法状态失败", err)
	}
	return nil
}

func mapAlgorithmReadChildren(children []read.Algorithm) []vo.AlgorithmVO {
	if len(children) == 0 {
		return []vo.AlgorithmVO{}
	}

	result := make([]vo.AlgorithmVO, 0, len(children))
	for _, child := range children {
		result = append(result, vo.AlgorithmVO{
			ID:          child.ID,
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


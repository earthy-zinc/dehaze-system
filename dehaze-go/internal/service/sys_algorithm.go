package service

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/global"
	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
)

// AlgorithmService 算法服务
type AlgorithmService struct {
	algorithmRepo repository.IAlgorithmRepository
}

// NewAlgorithmService 创建算法服务实例
func NewAlgorithmService(algorithmRepo repository.IAlgorithmRepository) *AlgorithmService {
	return &AlgorithmService{algorithmRepo: algorithmRepo}
}

// getRepo 获取 Repository（兼容零值实例）
func (s *AlgorithmService) getRepo() repository.IAlgorithmRepository {
	if s.algorithmRepo != nil {
		return s.algorithmRepo
	}
	return repository.NewAlgorithmRepository(global.DB)
}

// SetAlgorithmRepo 设置 Repository（测试用）
func (s *AlgorithmService) SetAlgorithmRepo(repo repository.IAlgorithmRepository) {
	s.algorithmRepo = repo
}

// ====================
// IAlgorithmService 接口实现
// ====================

// GetPage 算法分页列表
func (s *AlgorithmService) GetPage(ctx context.Context, q *query.AlgorithmQuery) (*vo.PageResult[vo.AlgorithmVO], error) {
	repo := s.getRepo()
	return repo.FindPage(ctx, q)
}

// GetOptions 获取算法下拉选项
func (s *AlgorithmService) GetOptions(ctx context.Context) ([]vo.Option, error) {
	repo := s.getRepo()
	return repo.FindOptions(ctx)
}

// GetFormData 获取算法表单数据
func (s *AlgorithmService) GetFormData(ctx context.Context, id int64) (*bo.AlgorithmFormBO, error) {
	repo := s.getRepo()

	algorithm, err := repo.FindByID(ctx, id)
	if err != nil {
		return nil, err
	}
	if algorithm == nil {
		return nil, errors.New("算法不存在")
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
	repo := s.getRepo()

	// 如果父节点ID不为0，检查父节点是否存在
	if form.ParentID != 0 {
		parentAlgorithm, err := repo.FindByID(ctx, form.ParentID)
		if err != nil {
			return err
		}
		if parentAlgorithm == nil {
			return errors.New("父算法不存在")
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

	return repo.Create(ctx, algorithm)
}

// Update 更新算法
func (s *AlgorithmService) Update(ctx context.Context, id int64, form *bo.AlgorithmFormBO) error {
	repo := s.getRepo()

	// 校验算法是否存在
	oldAlgorithm, err := repo.FindByID(ctx, id)
	if err != nil {
		return err
	}
	if oldAlgorithm == nil {
		return errors.New("算法不存在")
	}

	// 如果父节点ID不为0，检查父节点是否存在
	if form.ParentID != 0 && form.ParentID != id {
		parentAlgorithm, err := repo.FindByID(ctx, form.ParentID)
		if err != nil {
			return err
		}
		if parentAlgorithm == nil {
			return errors.New("父算法不存在")
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

	return repo.Update(ctx, oldAlgorithm)
}

// Delete 删除算法
func (s *AlgorithmService) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return errors.New("请选择要删除的算法")
	}

	// 检查是否有子算法
	var count int64
	err := global.DB.WithContext(ctx).Model(&model.SysAlgorithm{}).
		Where("parent_id IN ?", ids).
		Count(&count).Error
	if err != nil {
		return err
	}

	if count > 0 {
		return errors.New("存在子算法，无法删除")
	}

	repo := s.getRepo()
	return repo.Delete(ctx, ids)
}

// UpdateStatus 更新算法状态
func (s *AlgorithmService) UpdateStatus(ctx context.Context, id int64, status int8) error {
	repo := s.getRepo()
	return repo.UpdateStatus(ctx, id, status)
}

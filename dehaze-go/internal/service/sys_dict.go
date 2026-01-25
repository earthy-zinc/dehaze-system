package service

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/global"
	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
)

// DictService 字典数据服务
type DictService struct {
	dictRepo repository.IDictRepository
}

// NewDictService 创建字典数据服务实例
func NewDictService(dictRepo repository.IDictRepository) *DictService {
	return &DictService{dictRepo: dictRepo}
}

// getRepo 获取 Repository（兼容零值实例）
func (s *DictService) getRepo() repository.IDictRepository {
	if s.dictRepo != nil {
		return s.dictRepo
	}
	return repository.NewDictRepository(global.DB)
}

// SetDictRepo 设置 Repository（测试用）
func (s *DictService) SetDictRepo(repo repository.IDictRepository) {
	s.dictRepo = repo
}

// ====================
// IDictService 接口实现
// ====================

// GetPage 字典数据分页列表
func (s *DictService) GetPage(ctx context.Context, q *query.DictPageQuery) (*vo.PageResult[vo.DictPageVO], error) {
	repo := s.getRepo()
	return repo.FindPage(ctx, q)
}

// GetByTypeCode 根据类型编码获取字典列表
func (s *DictService) GetByTypeCode(ctx context.Context, typeCode string) ([]vo.Option, error) {
	repo := s.getRepo()

	dictList, err := repo.FindByTypeCode(ctx, typeCode)
	if err != nil {
		return nil, err
	}

	options := make([]vo.Option, 0, len(dictList))
	for _, dict := range dictList {
		options = append(options, vo.Option{
			Value: dict.Value,
			Label: dict.Name,
		})
	}

	return options, nil
}

// GetFormData 字典数据表单数据
func (s *DictService) GetFormData(ctx context.Context, id int64) (*bo.DictFormBO, error) {
	repo := s.getRepo()

	dict, err := repo.FindByID(ctx, id)
	if err != nil {
		return nil, err
	}
	if dict == nil {
		return nil, errors.New("字典数据项不存在")
	}

	form := &bo.DictFormBO{
		ID:       &dict.ID,
		TypeCode: dict.TypeCode,
		Name:     dict.Name,
		Value:    dict.Value,
		Status:   dict.Status,
		Sort:     dict.Sort,
		Remark:   dict.Remark,
	}

	return form, nil
}

// Create 创建字典
func (s *DictService) Create(ctx context.Context, form *bo.DictFormBO) error {
	repo := s.getRepo()

	dict := &model.SysDict{
		TypeCode:  form.TypeCode,
		Name:      form.Name,
		Value:     form.Value,
		Status:    form.Status,
		Sort:      form.Sort,
		Remark:    form.Remark,
		Defaulted: 0,
	}

	return repo.Create(ctx, dict)
}

// Update 更新字典
func (s *DictService) Update(ctx context.Context, id int64, form *bo.DictFormBO) error {
	repo := s.getRepo()

	dict, err := repo.FindByID(ctx, id)
	if err != nil {
		return err
	}
	if dict == nil {
		return errors.New("字典数据项不存在")
	}

	dict.TypeCode = form.TypeCode
	dict.Name = form.Name
	dict.Value = form.Value
	dict.Status = form.Status
	dict.Sort = form.Sort
	dict.Remark = form.Remark
	dict.UpdatedAt = time.Now()

	return repo.Update(ctx, dict)
}

// Delete 删除字典
func (s *DictService) Delete(ctx context.Context, ids []int64) error {
	repo := s.getRepo()
	if len(ids) == 0 {
		return errors.New("删除数据为空")
	}
	return repo.Delete(ctx, ids)
}

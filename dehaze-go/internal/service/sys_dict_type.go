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

// DictTypeService 字典类型服务
type DictTypeService struct {
	dictTypeRepo repository.IDictTypeRepository
}

// NewDictTypeService 创建字典类型服务实例
func NewDictTypeService(dictTypeRepo repository.IDictTypeRepository) *DictTypeService {
	return &DictTypeService{dictTypeRepo: dictTypeRepo}
}

// getRepo 获取 Repository（兼容零值实例）
func (s *DictTypeService) getRepo() repository.IDictTypeRepository {
	if s.dictTypeRepo != nil {
		return s.dictTypeRepo
	}
	return repository.NewDictTypeRepository(global.DB)
}

// SetDictTypeRepo 设置 Repository（测试用）
func (s *DictTypeService) SetDictTypeRepo(repo repository.IDictTypeRepository) {
	s.dictTypeRepo = repo
}

// ====================
// IDictTypeService 接口实现
// ====================

// GetPage 字典类型分页列表
func (s *DictTypeService) GetPage(ctx context.Context, q *query.DictTypePageQuery) (*vo.PageResult[vo.DictTypePageVO], error) {
	repo := s.getRepo()
	return repo.FindPage(ctx, q)
}

// GetFormData 字典类型表单数据
func (s *DictTypeService) GetFormData(ctx context.Context, id int64) (*bo.DictTypeFormBO, error) {
	repo := s.getRepo()

	dictType, err := repo.FindByID(ctx, id)
	if err != nil {
		return nil, err
	}
	if dictType == nil {
		return nil, errors.New("字典类型不存在")
	}

	form := &bo.DictTypeFormBO{
		ID:     &dictType.ID,
		Name:   dictType.Name,
		Code:   dictType.Code,
		Status: dictType.Status,
		Remark: dictType.Remark,
	}

	return form, nil
}

// Create 创建字典类型
func (s *DictTypeService) Create(ctx context.Context, form *bo.DictTypeFormBO) error {
	repo := s.getRepo()

	// 校验编码是否存在
	exists, err := repo.ExistsByCode(ctx, form.Code)
	if err != nil {
		return err
	}
	if exists {
		return errors.New("字典类型编码已存在")
	}

	dictType := &model.SysDictType{
		Name:   form.Name,
		Code:   form.Code,
		Status: form.Status,
		Remark: form.Remark,
	}

	return repo.Create(ctx, dictType)
}

// Update 更新字典类型
func (s *DictTypeService) Update(ctx context.Context, id int64, form *bo.DictTypeFormBO) error {
	repo := s.getRepo()

	// 校验字典类型是否存在
	oldDictType, err := repo.FindByID(ctx, id)
	if err != nil {
		return err
	}
	if oldDictType == nil {
		return errors.New("字典类型不存在")
	}

	// 校验编码是否存在（排除当前记录）
	if oldDictType.Code != form.Code {
		exists, err := repo.ExistsByCode(ctx, form.Code, id)
		if err != nil {
			return err
		}
		if exists {
			return errors.New("字典类型编码已存在")
		}
	}

	oldDictType.Name = form.Name
	oldDictType.Code = form.Code
	oldDictType.Status = form.Status
	oldDictType.Remark = form.Remark
	oldDictType.UpdatedAt = time.Now()

	err = repo.Update(ctx, oldDictType)
	if err != nil {
		return err
	}

	// 如果类型编码变化，需要同步更新字典数据的类型编码
	if oldDictType.Code != form.Code {
		dictRepo := repository.NewDictRepository(global.DB)
		var dictList []model.SysDict
		err = global.DB.Model(&model.SysDict{}).
			Where("type_code = ?", oldDictType.Code).
			Find(&dictList).Error
		if err != nil {
			return err
		}
		for _, dict := range dictList {
			dict.TypeCode = form.Code
			dict.UpdatedAt = time.Now()
			err = dictRepo.Update(ctx, &dict)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// Delete 删除字典类型
func (s *DictTypeService) Delete(ctx context.Context, ids []int64) error {
	repo := s.getRepo()
	if len(ids) == 0 {
		return errors.New("删除数据为空")
	}

	// 先查询要删除的字典类型编码
	var dictTypes []model.SysDictType
	err := global.DB.Model(&model.SysDictType{}).
		Where("id IN ?", ids).
		Select("code").
		Find(&dictTypes).Error
	if err != nil {
		return err
	}

	// 删除关联的字典数据项
	var dictTypeCodes []string
	for _, dictType := range dictTypes {
		dictTypeCodes = append(dictTypeCodes, dictType.Code)
	}
	if len(dictTypeCodes) > 0 {
		dictRepo := repository.NewDictRepository(global.DB)
		var dicts []model.SysDict
		err = global.DB.Model(&model.SysDict{}).
			Where("type_code IN ?", dictTypeCodes).
			Find(&dicts).Error
		if err != nil {
			return err
		}
		if len(dicts) > 0 {
			dictIDs := make([]int64, 0, len(dicts))
			for _, d := range dicts {
				dictIDs = append(dictIDs, d.ID)
			}
			err = dictRepo.Delete(ctx, dictIDs)
			if err != nil {
				return err
			}
		}
	}

	return repo.Delete(ctx, ids)
}

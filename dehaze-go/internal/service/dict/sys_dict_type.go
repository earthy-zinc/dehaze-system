package dict

import (
	"context"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	dictrepo "github.com/earthyzinc/dehaze-go/internal/repository/dict"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"

	"gorm.io/gorm"
)

// DictTypeService 字典类型服务
type DictTypeService struct {
	db           *gorm.DB
	dictTypeRepo dictrepo.IDictTypeRepository
	dictRepo     dictrepo.IDictRepository
	cache        types.ICache
}

// NewDictTypeService 创建字典类型服务实例
func NewDictTypeService(db *gorm.DB, dictTypeRepo dictrepo.IDictTypeRepository, dictRepo dictrepo.IDictRepository, cache types.ICache) *DictTypeService {
	return &DictTypeService{db: db, dictTypeRepo: dictTypeRepo, dictRepo: dictRepo, cache: cache}
}

// ====================
// IDictTypeService 接口实现
// ====================

// GetPage 字典类型分页列表
func (s *DictTypeService) GetPage(ctx context.Context, q *query.DictTypePageQuery) (*vo.PageResult[vo.DictTypePageVO], error) {
	readResult, err := s.dictTypeRepo.FindPage(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询字典类型分页列表失败", err)
	}
	if readResult == nil {
		return &vo.PageResult[vo.DictTypePageVO]{List: []vo.DictTypePageVO{}, Total: 0}, nil
	}

	voList := make([]vo.DictTypePageVO, 0, len(readResult.List))
	for _, item := range readResult.List {
		voList = append(voList, vo.DictTypePageVO{
			ID:         item.ID,
			Name:       item.Name,
			Code:       item.Code,
			Status:     item.Status,
			Remark:     item.Remark,
			CreateTime: item.CreateTime,
		})
	}

	return &vo.PageResult[vo.DictTypePageVO]{
		List:  voList,
		Total: readResult.Total,
	}, nil
}

// GetFormData 字典类型表单数据
func (s *DictTypeService) GetFormData(ctx context.Context, id int64) (*bo.DictTypeFormBO, error) {
	dictType, err := s.dictTypeRepo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询字典类型失败", err)
	}
	if dictType == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "字典类型不存在")
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
	// 校验编码是否存在
	exists, err := s.dictTypeRepo.ExistsByCode(ctx, form.Code)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "检查字典类型编码是否存在失败", err)
	}
	if exists {
		return common.NewBizError(common.DATA_EXISTS, "字典类型编码已存在")
	}

	dictType := &model.SysDictType{
		Name:   form.Name,
		Code:   form.Code,
		Status: form.Status,
		Remark: form.Remark,
	}

	if err := s.dictTypeRepo.Create(ctx, dictType); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "创建字典类型失败", err)
	}
	return nil
}

// Update 更新字典类型
func (s *DictTypeService) Update(ctx context.Context, id int64, form *bo.DictTypeFormBO) error {
	// 校验字典类型是否存在
	oldDictType, err := s.dictTypeRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询字典类型失败", err)
	}
	if oldDictType == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "字典类型不存在")
	}

	// 校验编码是否存在（排除当前记录）
	if oldDictType.Code != form.Code {
		exists, err := s.dictTypeRepo.ExistsByCode(ctx, form.Code, id)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "检查字典类型编码是否存在失败", err)
		}
		if exists {
			return common.NewBizError(common.DATA_EXISTS, "字典类型编码已存在")
		}
	}

	oldCode := oldDictType.Code
	needSyncCode := oldCode != form.Code

	// 如果不需要同步更新字典数据，直接更新
	if !needSyncCode {
		oldDictType.Name = form.Name
		oldDictType.Code = form.Code
		oldDictType.Status = form.Status
		oldDictType.Remark = form.Remark
		oldDictType.UpdatedAt = time.Now()

		if err := s.dictTypeRepo.Update(ctx, oldDictType); err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "更新字典类型失败", err)
		}
		return nil
	}

	// 使用事务处理更新操作（需要同时更新字典类型和字典数据的 type_code）
	err = s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		// 创建事务内的 Repository 实例
		txDictTypeRepo := s.dictTypeRepo.WithDB(tx)
		txDictRepo := s.dictRepo.WithDB(tx)

		// 更新字典类型
		oldDictType.Name = form.Name
		oldDictType.Code = form.Code
		oldDictType.Status = form.Status
		oldDictType.Remark = form.Remark
		oldDictType.UpdatedAt = time.Now()

		if err := txDictTypeRepo.Update(ctx, oldDictType); err != nil {
			return err
		}

		// 同步更新字典数据的类型编码
		if err := txDictRepo.UpdateTypeCode(ctx, oldCode, form.Code); err != nil {
			return err
		}

		return nil
	})

	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新字典类型失败", err)
	}

	// 清除缓存（新旧类型编码都需要清除）
	s.clearOptionsCache(ctx, oldCode)
	s.clearOptionsCache(ctx, form.Code)

	return nil
}

// Delete 删除字典类型
// force=true 时级联删除关联的字典数据，force=false 时存在关联数据则禁止删除
func (s *DictTypeService) Delete(ctx context.Context, ids []int64, force bool) error {
	if len(ids) == 0 {
		return common.NewBizError(common.PARAM_ERROR, "删除数据为空")
	}

	// 查询要删除的字典类型编码（同时用于校验字典类型是否存在）
	dictTypeCodes, err := s.dictTypeRepo.FindCodesByIDs(ctx, ids)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询字典类型编码失败", err)
	}
	// 校验字典类型是否存在（所有 ID 都不存在时返回错误）
	if len(dictTypeCodes) == 0 {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "字典类型不存在")
	}

	if len(dictTypeCodes) > 0 {
		if force {
			if err := s.dictRepo.DeleteByTypeCodes(ctx, dictTypeCodes); err != nil {
				return common.WrapBizError(common.DATABASE_ERROR, "级联删除字典数据失败", err)
			}
			for _, code := range dictTypeCodes {
				s.clearOptionsCache(ctx, code)
			}
		} else {
			count, err := s.dictRepo.CountByTypeCodes(ctx, dictTypeCodes)
			if err != nil {
				return common.WrapBizError(common.DATABASE_ERROR, "检查关联字典数据失败", err)
			}
			if count > 0 {
				return common.NewBizError(common.DATA_BIND_EXISTS, "存在关联的字典数据，无法删除")
			}
		}
	}

	// 执行删除
	if err := s.dictTypeRepo.Delete(ctx, ids); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除字典类型失败", err)
	}
	return nil
}

// clearOptionsCache 清除字典下拉选项缓存
func (s *DictTypeService) clearOptionsCache(ctx context.Context, typeCode string) {
	if s.cache == nil || typeCode == "" {
		return
	}
	cacheKey := fmt.Sprintf("%s%s", DictOptionsCachePrefix, typeCode)
	_ = s.cache.Delete(ctx, cacheKey)
}

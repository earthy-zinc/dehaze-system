package dict

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	dictrepo "github.com/earthyzinc/dehaze-go/internal/repository/dict"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

const (
	// DictOptionsCachePrefix 字典下拉选项缓存前缀
	DictOptionsCachePrefix = "dict:options:"
	// DictOptionsCacheTTL 字典下拉选项缓存过期时间（1小时）
	DictOptionsCacheTTL = time.Hour
)

// DictService 字典数据服务
type DictService struct {
	dictRepo     dictrepo.IDictRepository
	dictTypeRepo dictrepo.IDictTypeRepository
	cache        types.ICache
}

// NewDictService 创建字典数据服务实例
func NewDictService(dictRepo dictrepo.IDictRepository, dictTypeRepo dictrepo.IDictTypeRepository, cache types.ICache) *DictService {
	return &DictService{
		dictRepo:     dictRepo,
		dictTypeRepo: dictTypeRepo,
		cache:        cache,
	}
}

// ====================
// IDictService 接口实现
// ====================

// GetPage 字典数据分页列表
func (s *DictService) GetPage(ctx context.Context, q *query.DictPageQuery) (*vo.PageResult[vo.DictPageVO], error) {
	readResult, err := s.dictRepo.FindPage(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询字典数据分页列表失败", err)
	}
	if readResult == nil {
		return &vo.PageResult[vo.DictPageVO]{List: []vo.DictPageVO{}, Total: 0}, nil
	}

	voList := make([]vo.DictPageVO, 0, len(readResult.List))
	for _, item := range readResult.List {
		voList = append(voList, vo.DictPageVO{
			ID:         item.ID,
			Name:       item.Name,
			Value:      item.Value,
			TypeCode:   item.TypeCode,
			Defaulted:  item.Defaulted,
			Sort:       item.Sort,
			Status:     item.Status,
			Remark:     item.Remark,
			CreateTime: item.CreateTime,
		})
	}

	return &vo.PageResult[vo.DictPageVO]{
		List:  voList,
		Total: readResult.Total,
	}, nil
}

// GetByTypeCode 根据类型编码获取字典列表（下拉选项）
// 根据文档要求：过滤禁用状态，使用缓存
func (s *DictService) GetByTypeCode(ctx context.Context, typeCode string) ([]vo.Option, error) {
	// 参数校验
	if typeCode == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "类型编码不能为空")
	}

	cacheKey := DictOptionsCachePrefix + typeCode

	// 尝试从缓存获取
	if s.cache != nil {
		cachedData, err := s.cache.Get(ctx, cacheKey)
		if err == nil && cachedData != "" {
			var options []vo.Option
			if err := json.Unmarshal([]byte(cachedData), &options); err == nil {
				return options, nil
			}
		}
	}

	// 从数据库查询
	dictList, err := s.dictRepo.FindByTypeCode(ctx, typeCode)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "根据类型编码查询字典列表失败", err)
	}

	// 过滤禁用状态，只返回启用状态的数据
	options := make([]vo.Option, 0, len(dictList))
	for _, dict := range dictList {
		if dict.Status == 1 { // 只返回启用状态的数据
			options = append(options, vo.Option{
				Value: dict.Value,
				Label: dict.Name,
			})
		}
	}

	// 写入缓存
	if s.cache != nil {
		if data, err := json.Marshal(options); err == nil {
			_ = s.cache.Set(ctx, cacheKey, string(data), DictOptionsCacheTTL)
		}
	}

	return options, nil
}

// GetFormData 字典数据表单数据
func (s *DictService) GetFormData(ctx context.Context, id int64) (*bo.DictFormBO, error) {
	dict, err := s.dictRepo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询字典数据失败", err)
	}
	if dict == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "字典数据项不存在")
	}

	form := &bo.DictFormBO{
		ID:        &dict.ID,
		TypeCode:  dict.TypeCode,
		Name:      dict.Name,
		Value:     dict.Value,
		Status:    dict.Status,
		Sort:      dict.Sort,
		Remark:    dict.Remark,
		Defaulted: dict.Defaulted,
	}

	return form, nil
}

// Create 创建字典
// 根据文档要求：校验类型编码有效性、字典值唯一性
func (s *DictService) Create(ctx context.Context, form *bo.DictFormBO) error {
	// 校验类型编码有效性
	dictType, err := s.dictTypeRepo.FindByCode(ctx, form.TypeCode)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "检查字典类型编码失败", err)
	}
	if dictType == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "字典类型编码不存在")
	}

	// 校验字典值唯一性（同类型下）
	exists, err := s.dictRepo.ExistsByTypeCodeAndValue(ctx, form.TypeCode, form.Value)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "检查字典值唯一性失败", err)
	}
	if exists {
		return common.NewBizError(common.DATA_EXISTS, "该类型下字典值已存在")
	}

	dict := &model.SysDict{
		TypeCode:  form.TypeCode,
		Name:      form.Name,
		Value:     form.Value,
		Status:    form.Status,
		Sort:      form.Sort,
		Remark:    form.Remark,
		Defaulted: form.Defaulted,
	}

	if err := s.dictRepo.Create(ctx, dict); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "创建字典数据失败", err)
	}

	// 清除相关缓存
	s.clearOptionsCache(ctx, form.TypeCode)

	return nil
}

// Update 更新字典
// 根据文档要求：校验字典值唯一性、清除缓存
func (s *DictService) Update(ctx context.Context, id int64, form *bo.DictFormBO) error {
	dict, err := s.dictRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询字典数据失败", err)
	}
	if dict == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "字典数据项不存在")
	}

	// 校验类型编码有效性
	dictType, err := s.dictTypeRepo.FindByCode(ctx, form.TypeCode)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "检查字典类型编码失败", err)
	}
	if dictType == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "字典类型编码不存在")
	}

	// 校验字典值唯一性（同类型下，排除当前记录）
	if dict.Value != form.Value || dict.TypeCode != form.TypeCode {
		exists, err := s.dictRepo.ExistsByTypeCodeAndValue(ctx, form.TypeCode, form.Value, id)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "检查字典值唯一性失败", err)
		}
		if exists {
			return common.NewBizError(common.DATA_EXISTS, "该类型下字典值已存在")
		}
	}

	// 记录旧的类型编码用于清除缓存
	oldTypeCode := dict.TypeCode

	dict.TypeCode = form.TypeCode
	dict.Name = form.Name
	dict.Value = form.Value
	dict.Status = form.Status
	dict.Sort = form.Sort
	dict.Remark = form.Remark
	dict.Defaulted = form.Defaulted
	dict.UpdatedAt = time.Now()

	if err := s.dictRepo.Update(ctx, dict); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新字典数据失败", err)
	}

	// 清除相关缓存（新旧类型编码都需要清除）
	s.clearOptionsCache(ctx, oldTypeCode)
	if oldTypeCode != form.TypeCode {
		s.clearOptionsCache(ctx, form.TypeCode)
	}

	return nil
}

// Delete 删除字典
// 根据文档要求：清除缓存
func (s *DictService) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return common.NewBizError(common.PARAM_ERROR, "删除数据为空")
	}

	// 查询要删除的字典数据，收集类型编码用于清除缓存
	dictList, err := s.findDictsByIDs(ctx, ids)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询字典数据失败", err)
	}

	// 收集需要清除缓存的类型编码
	typeCodeSet := make(map[string]struct{})
	for _, dict := range dictList {
		typeCodeSet[dict.TypeCode] = struct{}{}
	}

	if err := s.dictRepo.Delete(ctx, ids); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除字典数据失败", err)
	}

	// 清除相关缓存
	for typeCode := range typeCodeSet {
		s.clearOptionsCache(ctx, typeCode)
	}

	return nil
}

// findDictsByIDs 根据 ID 列表查询字典列表
func (s *DictService) findDictsByIDs(ctx context.Context, ids []int64) ([]model.SysDict, error) {
	return s.dictRepo.FindByIDs(ctx, ids)
}

// clearOptionsCache 清除字典下拉选项缓存
func (s *DictService) clearOptionsCache(ctx context.Context, typeCode string) {
	if s.cache == nil || typeCode == "" {
		return
	}
	cacheKey := fmt.Sprintf("%s%s", DictOptionsCachePrefix, typeCode)
	_ = s.cache.Delete(ctx, cacheKey)
}

package dict

import (
	"context"
	"encoding/json"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	dictrepo "github.com/earthyzinc/dehaze-go/internal/repository/dict"
	"github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

const (
	// DictOptionsCachePrefix 字典下拉选项缓存前缀（java/go 下拉选项读路径）
	DictOptionsCachePrefix = "dict:data:"
	// DictValueCachePrefix 字典业务值缓存前缀：`dict:value:{typeCode}:{key}`。
	//
	// **跨端共享键**——python 的字典值读路径（`dict_service.get_dict_value`，平台默认模型
	// 解析、会员成长值规则、收藏容量等）走该前缀；java/go 改字典后只失效 `dict:data:`
	// 会让 python 侧判定滞后到 TTL（1h）到期。两者必须一并失效。
	DictValueCachePrefix = "dict:value:"
	// DictOptionsCacheTTL 字典下拉选项缓存过期时间（1小时）
	DictOptionsCacheTTL = time.Hour

	// cacheInvalidationChannel 缓存失效广播频道（三端共用；python 订阅后清进程内 L1）
	cacheInvalidationChannel = "cache:invalidation"
	// dictCacheSenderID 广播发送者标识（接收端据此过滤自身消息）
	dictCacheSenderID = "dehaze-go"
	// dictValueScanCount 通配失效单次 SCAN 的 COUNT
	dictValueScanCount = 100
)

// clearDictCaches 字典写路径统一失效：下拉选项缓存 + 业务值缓存（与 python 同一批键）。
//
// 两处都带跨实例广播：`dict:data:{typeCode}` 经多级缓存 Delete 发 "key" 消息；
// `dict:value:{typeCode}:*` 是通配失效，发 "pattern" 消息（python 端按 pattern 清 L1）。
func clearDictCaches(ctx context.Context, cache types.ICache, typeCode string) {
	if typeCode == "" {
		return
	}
	if cache != nil {
		_ = cache.Delete(ctx, DictOptionsCachePrefix+typeCode)
	}
	invalidateDictValueCache(ctx, typeCode)
}

// invalidateDictValueCache 失效 `dict:value:{typeCode}:*`：L2 逐键删除 + 广播 pattern 清 L1。
//
// 键由 python 写入（本端不读该前缀），故本端只需清共享 L2 并广播；无 Redis 时静默跳过
// 会让调用方误判为已失效，所以异常一律显式 WARN 留痕。
func invalidateDictValueCache(ctx context.Context, typeCode string) {
	client := redis.GetClient()
	if client == nil {
		logger.Warn("Redis 未初始化，跳过字典值缓存失效", zap.String("typeCode", typeCode))
		return
	}
	pattern := DictValueCachePrefix + typeCode + ":*"

	var cursor uint64
	for {
		keys, next, err := client.Scan(ctx, cursor, pattern, dictValueScanCount).Result()
		if err != nil {
			logger.Warn("扫描字典值缓存失败", zap.String("pattern", pattern), zap.Error(err))
			break
		}
		if len(keys) > 0 {
			if err := client.Del(ctx, keys...).Err(); err != nil {
				logger.Warn("删除字典值缓存失败", zap.Strings("keys", keys), zap.Error(err))
			}
		}
		cursor = next
		if cursor == 0 {
			break
		}
	}

	payload, err := json.Marshal(map[string]any{
		"type":     "pattern",
		"key":      pattern,
		"senderId": dictCacheSenderID,
	})
	if err != nil {
		logger.Warn("序列化字典值缓存失效消息失败", zap.String("pattern", pattern), zap.Error(err))
		return
	}
	if err := client.Publish(ctx, cacheInvalidationChannel, payload).Err(); err != nil {
		logger.Warn("广播字典值缓存失效失败", zap.String("pattern", pattern), zap.Error(err))
	}
}

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

	status := dict.Status
	sort := dict.Sort
	form := &bo.DictFormBO{
		ID:        &dict.ID,
		TypeCode:  dict.TypeCode,
		Name:      dict.Name,
		Value:     dict.Value,
		Status:    &status,
		Sort:      &sort,
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

	status := int8(1)
	if form.Status != nil {
		status = *form.Status
	}
	sort := 1
	if form.Sort != nil {
		sort = *form.Sort
	}

	dict := &model.SysDict{
		TypeCode:  form.TypeCode,
		Name:      form.Name,
		Value:     form.Value,
		Status:    status,
		Sort:      sort,
		Remark:    form.Remark,
		Defaulted: form.Defaulted,
	}

	if err := s.dictRepo.Create(ctx, dict); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "创建字典数据失败", err)
	}

	// 清除相关缓存（下拉选项 + python 业务值缓存）
	clearDictCaches(ctx, s.cache, form.TypeCode)

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

	// 校验字典值唯一性（同类型下，排除当前记录，查全表含软删行）— typeCode 只读，使用原记录的 typeCode
	if dict.Value != form.Value {
		exists, err := s.dictRepo.ExistsByTypeCodeAndValue(ctx, dict.TypeCode, form.Value, id)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "检查字典值唯一性失败", err)
		}
		if exists {
			return common.NewBizError(common.DATA_EXISTS, "该类型下字典值已存在")
		}
	}

	// 更新字典数据（typeCode 只读，不更新；未提供的字段保留原值）
	dict.Name = form.Name
	dict.Value = form.Value
	if form.Status != nil {
		dict.Status = *form.Status
	}
	if form.Sort != nil {
		dict.Sort = *form.Sort
	}
	dict.Remark = form.Remark
	dict.Defaulted = form.Defaulted
	// 时间截断到秒：列为 DATETIME（秒精度），直写带纳秒的 time.Now() 会被 MySQL 进位成下一刻
	dict.UpdatedAt = time.Now().Truncate(time.Second)

	if err := s.dictRepo.Update(ctx, dict); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新字典数据失败", err)
	}

	// 清除缓存（typeCode 不变，只需清除一个）
	clearDictCaches(ctx, s.cache, dict.TypeCode)

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
	// 校验字典数据是否存在（所有 ID 都不存在时返回错误）
	if len(dictList) == 0 {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "字典数据项不存在")
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
		clearDictCaches(ctx, s.cache, typeCode)
	}

	return nil
}

// findDictsByIDs 根据 ID 列表查询字典列表
func (s *DictService) findDictsByIDs(ctx context.Context, ids []int64) ([]model.SysDict, error) {
	return s.dictRepo.FindByIDs(ctx, ids)
}

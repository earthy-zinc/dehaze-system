package ai

import (
	"context"
	"encoding/json"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

// ProviderService AI 模型供应商与 API Key 管理
type ProviderService struct {
	providerRepo *airepo.ProviderRepository
	keyRepo      *airepo.ProviderKeyRepository
	health       *HealthService
}

func NewProviderService(
	providerRepo *airepo.ProviderRepository,
	keyRepo *airepo.ProviderKeyRepository,
	health *HealthService,
) *ProviderService {
	return &ProviderService{providerRepo: providerRepo, keyRepo: keyRepo, health: health}
}

func (s *ProviderService) ListProviders(ctx context.Context, q *bo.ProviderQuery) (*vo.PageResult[vo.ProviderVO], error) {
	page, size := q.PageNum, q.PageSize
	providers, total, err := s.providerRepo.PaginateProviders(ctx, page, size, q.Keyword)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询供应商列表失败", err)
	}
	items := make([]vo.ProviderVO, 0, len(providers))
	for i := range providers {
		item := toProviderVO(&providers[i])
		health := s.health.Status(ctx, providers[i].ID)
		item.Health = &health
		items = append(items, item)
	}
	return &vo.PageResult[vo.ProviderVO]{List: items, Total: total}, nil
}

// ListEnabledProviders 启用供应商精简列表（缓存 ai:provider:list TTL 1h）
func (s *ProviderService) ListEnabledProviders(ctx context.Context) ([]vo.ProviderEnabledVO, error) {
	if client := redisClient(); client != nil {
		if raw, err := client.Get(ctx, providerListCacheKey).Bytes(); err == nil && len(raw) > 0 {
			var cached []providerCacheDTO
			if json.Unmarshal(raw, &cached) == nil {
				items := make([]vo.ProviderEnabledVO, 0, len(cached))
				for i := range cached {
					items = append(items, providerFromCache(&cached[i]))
				}
				return items, nil
			}
		}
	}
	providers, err := s.providerRepo.ListEnabled(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询启用供应商失败", err)
	}
	items := make([]vo.ProviderEnabledVO, 0, len(providers))
	for i := range providers {
		health := s.health.Status(ctx, providers[i].ID)
		items = append(items, vo.ProviderEnabledVO{
			ID:           providers[i].ID,
			ProviderCode: providers[i].ProviderCode,
			DisplayName:  providers[i].DisplayName,
			ProtocolType: providers[i].ProtocolType,
			Health:       &health,
			Status:       providers[i].Status,
		})
	}
	if client := redisClient(); client != nil {
		cachePayload := make([]providerCacheDTO, 0, len(items))
		for i := range items {
			cachePayload = append(cachePayload, providerToCache(&items[i]))
		}
		if b, err := json.Marshal(cachePayload); err == nil {
			client.Set(ctx, providerListCacheKey, b, cacheTTLHour)
		}
	}
	return items, nil
}

// providerCacheDTO 供应商精简列表共享缓存（ai:provider:list）传输对象：键名一律 snake_case，
// 与 python（ProviderEnabledResult.model_dump(mode="json")）和 java（AiJsonUtils SNAKE_MAPPER）互认；
// 与对外 camelCase 的 vo.ProviderEnabledVO 严格隔离。
type providerCacheDTO struct {
	ID           int64   `json:"id"`
	ProviderCode string  `json:"provider_code"`
	DisplayName  string  `json:"display_name"`
	ProtocolType string  `json:"protocol_type"`
	Health       *string `json:"health"`
	Status       int8    `json:"status"`
}

func providerToCache(item *vo.ProviderEnabledVO) providerCacheDTO {
	return providerCacheDTO{
		ID: item.ID, ProviderCode: item.ProviderCode, DisplayName: item.DisplayName,
		ProtocolType: item.ProtocolType, Health: item.Health, Status: item.Status,
	}
}

func providerFromCache(dto *providerCacheDTO) vo.ProviderEnabledVO {
	return vo.ProviderEnabledVO{
		ID: dto.ID, ProviderCode: dto.ProviderCode, DisplayName: dto.DisplayName,
		ProtocolType: dto.ProtocolType, Health: dto.Health, Status: dto.Status,
	}
}

func (s *ProviderService) CreateProvider(ctx context.Context, form *bo.ProviderCreateForm, operatorID int64) (*vo.ProviderVO, error) {
	existing, err := s.providerRepo.GetByCode(ctx, form.ProviderCode, true)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "校验供应商编码失败", err)
	}
	if existing != nil {
		if existing.Deleted > 0 {
			return nil, common.NewBizError(common.DATA_EXISTS, "供应商编码已被历史记录占用，不可复用")
		}
		return nil, common.NewBizError(common.DATA_EXISTS, "供应商编码已存在")
	}

	p := &model.SysAiProvider{
		ProviderCode:        form.ProviderCode,
		DisplayName:         form.DisplayName,
		ApiBaseUrl:          form.ApiBaseUrl,
		ProtocolType:        protocolOr(form.ProtocolType),
		AuthType:            authOr(form.AuthType),
		DefaultHeaders:      form.DefaultHeaders,
		SortOrder:           intOr(form.SortOrder, 0),
		HealthCheckEnabled:  int8Or(form.HealthCheckEnabled, 1),
		UserIdentityForward: form.UserIdentityForward,
		Remark:              nil,
		Status:              int8Or(form.Status, 1),
	}
	if provided, isNull := rawProvided(form.Remark); provided && !isNull {
		value, err := jsonString(form.Remark)
		if err != nil {
			return nil, common.NewBizError(common.PARAM_ERROR, "remark 格式不正确")
		}
		p.Remark = &value
	}
	p.CreateBy = operatorID
	p.UpdateBy = operatorID

	if err := s.providerRepo.Create(ctx, p); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建供应商失败", err)
	}
	clearProviderCache(ctx)
	s.health.SetHealthCheckEnabled(ctx, p.ID, p.HealthCheckEnabled == 1)
	result := toProviderVO(p)
	return &result, nil
}

func (s *ProviderService) UpdateProvider(ctx context.Context, providerID int64, form *bo.ProviderUpdateForm, operatorID int64) (*vo.ProviderVO, error) {
	p, err := s.providerRepo.GetByID(ctx, providerID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询供应商失败", err)
	}
	if p == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "供应商不存在")
	}

	updates := map[string]interface{}{"update_by": operatorID}
	if form.DisplayName != nil {
		updates["display_name"] = *form.DisplayName
	}
	if form.ApiBaseUrl != nil {
		updates["api_base_url"] = *form.ApiBaseUrl
	}
	if form.ProtocolType != nil {
		updates["protocol_type"] = *form.ProtocolType
	}
	if form.AuthType != nil {
		updates["auth_type"] = *form.AuthType
	}
	if form.SortOrder != nil {
		updates["sort_order"] = *form.SortOrder
	}
	if form.HealthCheckEnabled != nil {
		updates["health_check_enabled"] = *form.HealthCheckEnabled
	}
	if form.Status != nil {
		updates["status"] = *form.Status
	}
	if provided, isNull := rawProvided(form.DefaultHeaders); provided {
		updates["default_headers"] = jsonColumnValue(form.DefaultHeaders, isNull)
	}
	if provided, isNull := rawProvided(form.UserIdentityForward); provided {
		updates["user_identity_forward"] = jsonColumnValue(form.UserIdentityForward, isNull)
	}
	if provided, isNull := rawProvided(form.Remark); provided {
		if isNull {
			updates["remark"] = nil
		} else {
			value, convErr := jsonString(form.Remark)
			if convErr != nil {
				return nil, common.NewBizError(common.PARAM_ERROR, "remark 格式不正确")
			}
			updates["remark"] = value
		}
	}

	if err := s.providerRepo.Update(ctx, p.ID, updates); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "更新供应商失败", err)
	}
	clearProviderCache(ctx)

	updated, err := s.providerRepo.GetByID(ctx, p.ID)
	if err != nil || updated == nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询供应商失败", err)
	}
	s.health.SetHealthCheckEnabled(ctx, updated.ID, updated.HealthCheckEnabled == 1)
	result := toProviderVO(updated)
	return &result, nil
}

func (s *ProviderService) DeleteProvider(ctx context.Context, providerID, operatorID int64) error {
	p, err := s.providerRepo.GetByID(ctx, providerID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询供应商失败", err)
	}
	if p == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "供应商不存在")
	}
	bound, err := s.providerRepo.CountModels(ctx, providerID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "统计关联模型失败", err)
	}
	if bound > 0 {
		return common.NewBizError(common.DATA_BIND_EXISTS, "存在模型引用该供应商（含禁用模型），请先删除或转移关联模型")
	}
	if err := s.providerRepo.SoftDelete(ctx, providerID, operatorID); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除供应商失败", err)
	}
	clearProviderCache(ctx)
	s.health.ClearProviderHealth(ctx, providerID)
	return nil
}

// ==================== API Key ====================

func (s *ProviderService) ListProviderKeys(ctx context.Context, providerID int64) ([]vo.ProviderKeyVO, error) {
	if err := s.requireProvider(ctx, providerID); err != nil {
		return nil, err
	}
	keys, err := s.keyRepo.ListByProvider(ctx, providerID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询API Key失败", err)
	}
	items := make([]vo.ProviderKeyVO, 0, len(keys))
	for i := range keys {
		items = append(items, toProviderKeyVO(&keys[i]))
	}
	return items, nil
}

func (s *ProviderService) CreateProviderKey(ctx context.Context, providerID int64, form *bo.ProviderKeyCreateForm, operatorID int64) (*vo.ProviderKeyVO, error) {
	if err := s.requireProvider(ctx, providerID); err != nil {
		return nil, err
	}
	hash := HashSecret(form.Key)
	existing, err := s.keyRepo.GetByHash(ctx, hash)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "校验API Key失败", err)
	}
	if existing != nil {
		return nil, common.NewBizError(common.DATA_EXISTS, "该 API Key 已存在")
	}
	cipherText, err := EncryptSecret(form.Key)
	if err != nil {
		return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "API Key 加密失败", err)
	}
	prefix := MaskSecret(form.Key)
	k := &model.SysAiProviderKey{
		ProviderID: providerID,
		Name:       form.Name,
		KeyHash:    hash,
		KeyPrefix:  &prefix,
		KeyCipher:  cipherText,
		Status:     int8Or(form.Status, 1),
		Priority:   intOr(form.Priority, 0),
		Weight:     intOr(form.Weight, 1),
		DailyQuota: form.DailyQuota,
		RpmLimit:   form.RpmLimit,
		ExpiresAt:  form.ExpiresAt,
	}
	k.CreateBy = operatorID
	k.UpdateBy = operatorID
	if err := s.keyRepo.Create(ctx, k); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建API Key失败", err)
	}
	result := toProviderKeyVO(k)
	return &result, nil
}

func (s *ProviderService) UpdateProviderKey(ctx context.Context, providerID, keyID int64, form *bo.ProviderKeyUpdateForm, operatorID int64) (*vo.ProviderKeyVO, error) {
	k, err := s.requireKey(ctx, providerID, keyID)
	if err != nil {
		return nil, err
	}
	updates := map[string]interface{}{"update_by": operatorID}
	if form.Name != nil {
		updates["name"] = *form.Name
	}
	if form.Priority != nil {
		updates["priority"] = *form.Priority
	}
	if form.Weight != nil {
		updates["weight"] = *form.Weight
	}
	if form.Status != nil {
		updates["status"] = *form.Status
	}
	if form.DailyQuota != nil {
		updates["daily_quota"] = *form.DailyQuota
	}
	if form.RpmLimit != nil {
		updates["rpm_limit"] = *form.RpmLimit
	}
	if form.ExpiresAt != nil {
		updates["expires_at"] = *form.ExpiresAt
	}
	if err := s.keyRepo.Update(ctx, k.ID, updates); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "更新API Key失败", err)
	}
	updated, err := s.keyRepo.GetByID(ctx, k.ID)
	if err != nil || updated == nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询API Key失败", err)
	}
	result := toProviderKeyVO(updated)
	return &result, nil
}

// DeleteProviderKey 删除 Key；唯一启用 Key 不可删除（防供应商无 Key 可用）
func (s *ProviderService) DeleteProviderKey(ctx context.Context, providerID, keyID int64) error {
	k, err := s.requireKey(ctx, providerID, keyID)
	if err != nil {
		return err
	}
	if k.Status == 1 {
		enabled, countErr := s.keyRepo.CountEnabledByProvider(ctx, providerID)
		if countErr != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "统计启用API Key失败", countErr)
		}
		if enabled <= 1 {
			return common.NewBizError(common.OPERATION_NOT_ALLOW,
				"该供应商唯一启用 Key，不可删除，请先新增其他 Key 或禁用后再删除")
		}
	}
	if err := s.keyRepo.DeleteByID(ctx, keyID); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除API Key失败", err)
	}
	return nil
}

// CloseProviderCircuit 手动解除供应商熔断（直接操作 python 同款 Redis 熔断键）
func (s *ProviderService) CloseProviderCircuit(ctx context.Context, providerID int64) error {
	if err := s.requireProvider(ctx, providerID); err != nil {
		return err
	}
	s.health.CloseCircuit(ctx, providerID)
	return nil
}

func (s *ProviderService) requireProvider(ctx context.Context, providerID int64) error {
	p, err := s.providerRepo.GetByID(ctx, providerID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询供应商失败", err)
	}
	if p == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "供应商不存在")
	}
	return nil
}

func (s *ProviderService) requireKey(ctx context.Context, providerID, keyID int64) (*model.SysAiProviderKey, error) {
	k, err := s.keyRepo.GetByID(ctx, keyID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询API Key失败", err)
	}
	if k == nil || k.ProviderID != providerID {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "API Key 不存在")
	}
	return k, nil
}

func toProviderVO(p *model.SysAiProvider) vo.ProviderVO {
	return vo.ProviderVO{
		ID:                  p.ID,
		ProviderCode:        p.ProviderCode,
		DisplayName:         p.DisplayName,
		ApiBaseUrl:          p.ApiBaseUrl,
		ProtocolType:        p.ProtocolType,
		AuthType:            p.AuthType,
		DefaultHeaders:      p.DefaultHeaders,
		SortOrder:           p.SortOrder,
		HealthCheckEnabled:  p.HealthCheckEnabled,
		UserIdentityForward: p.UserIdentityForward,
		Remark:              p.Remark,
		Status:              p.Status,
		CreateTime:          p.CreatedAt,
		UpdateTime:          p.UpdatedAt,
	}
}

func toProviderKeyVO(k *model.SysAiProviderKey) vo.ProviderKeyVO {
	return vo.ProviderKeyVO{
		ID:         k.ID,
		ProviderID: k.ProviderID,
		Name:       k.Name,
		KeyPrefix:  k.KeyPrefix,
		Status:     k.Status,
		Priority:   k.Priority,
		Weight:     k.Weight,
		DailyQuota: k.DailyQuota,
		RpmLimit:   k.RpmLimit,
		ExpiresAt:  k.ExpiresAt,
		LastUsedAt: k.LastUsedAt,
		LastUsedBy: k.LastUsedBy,
		CreateTime: k.CreatedAt,
		UpdateTime: k.UpdatedAt,
	}
}

func protocolOr(v string) string {
	if v == "" {
		return "openai_compat"
	}
	return v
}

func authOr(v string) string {
	if v == "" {
		return "bearer"
	}
	return v
}

// jsonColumnValue 显式 null 落 NULL，否则原样写入 JSON 文本
func jsonColumnValue(raw json.RawMessage, isNull bool) interface{} {
	if isNull {
		return nil
	}
	return raw
}

func jsonString(raw json.RawMessage) (string, error) {
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", err
	}
	return value, nil
}

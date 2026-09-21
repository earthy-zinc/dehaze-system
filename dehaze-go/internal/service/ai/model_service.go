package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"gorm.io/gorm"
)

// MessageSender 站内信发送（模型下线通知），由 app 注入 message 服务
type MessageSender interface {
	Send(ctx context.Context, form *bo.MessageSendForm) (*vo.MessageSendResultVO, error)
}

// ModelService AI 模型注册表与用户售价管理
type ModelService struct {
	db        *gorm.DB
	modelRepo *airepo.ModelRepository
	priceRepo *airepo.ModelPriceRepository
	health    *HealthService
	notifier  MessageSender
}

func NewModelService(
	db *gorm.DB,
	modelRepo *airepo.ModelRepository,
	priceRepo *airepo.ModelPriceRepository,
	health *HealthService,
	notifier MessageSender,
) *ModelService {
	return &ModelService{db: db, modelRepo: modelRepo, priceRepo: priceRepo, health: health, notifier: notifier}
}

// ListModels 模型分页列表（含近 24h 真实调用统计）
func (s *ModelService) ListModels(ctx context.Context, q *bo.AiModelQuery) (*vo.PageResult[vo.AiModelVO], error) {
	// 列表筛选同为 Literal 约束：非法值在 python 是请求校验阶段的 A0400（而非"过滤后空列表"），
	// 空串视为不筛选（显式空串与不传在 go 侧不可区分，沿用既有空串差异登记口径）
	if q.ModelType != "" {
		if err := ValidateModelType(q.ModelType); err != nil {
			return nil, err
		}
	}
	page, size := q.PageNum, q.PageSize
	models, total, err := s.modelRepo.PaginateModels(ctx, page, size, q.Keyword, q.ModelType)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询模型列表失败", err)
	}
	stats, err := s.usageStats24h(ctx, models)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询模型调用统计失败", err)
	}
	items := make([]vo.AiModelVO, 0, len(models))
	for i := range models {
		item := toAiModelVO(&models[i])
		if stat, ok := stats[models[i].ID]; ok {
			item.Calls24h = stat.Calls24h
			item.SuccessRate24h = stat.SuccessRate24h
			item.LastCallAt = stat.LastCallAt
		}
		items = append(items, item)
	}
	return &vo.PageResult[vo.AiModelVO]{List: items, Total: total}, nil
}

// ListEnabledModels 启用模型列表（登录用户可见，按 VIP 等级过滤；缓存 ai:model:list TTL 1h）
func (s *ModelService) ListEnabledModels(ctx context.Context, userID int64, modelType string) ([]vo.AiModelVO, error) {
	cached, err := s.loadEnabledModelCache(ctx)
	if err != nil {
		return nil, err
	}
	level := s.health.UserLevel(ctx, userID)
	items := make([]vo.AiModelVO, 0, len(cached))
	for _, item := range cached {
		if item.VipLevel > int8(level) {
			continue
		}
		if modelType != "" && item.ModelType != modelType {
			continue
		}
		items = append(items, item)
	}
	return items, nil
}

func (s *ModelService) loadEnabledModelCache(ctx context.Context) ([]vo.AiModelVO, error) {
	if client := redisClient(); client != nil {
		if raw, err := client.Get(ctx, modelListCacheKey).Bytes(); err == nil && len(raw) > 0 {
			var cached []modelCacheDTO
			if json.Unmarshal(raw, &cached) == nil {
				items := make([]vo.AiModelVO, 0, len(cached))
				for i := range cached {
					items = append(items, modelFromCache(&cached[i]))
				}
				return items, nil
			}
		}
	}
	// 缓存保存全部启用模型快照（不按类型过滤），类型筛选在读缓存后内存过滤，避免互相污染
	models, err := s.modelRepo.ListEnabled(ctx, "")
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询启用模型失败", err)
	}
	fallbackTargets := make(map[int64]bool, len(models))
	for i := range models {
		if models[i].FallbackModelID != nil {
			fallbackTargets[*models[i].FallbackModelID] = true
		}
	}
	snapshot := make([]vo.AiModelVO, 0, len(models))
	for i := range models {
		item := toAiModelVO(&models[i])
		item.SpeedTier = speedTierOf(s.health.Snapshot(ctx, models[i].ProviderID))
		item.IsFallbackTarget = fallbackTargets[models[i].ID]
		snapshot = append(snapshot, item)
	}
	if client := redisClient(); client != nil {
		cachePayload := make([]modelCacheDTO, 0, len(snapshot))
		for i := range snapshot {
			cachePayload = append(cachePayload, modelToCache(&snapshot[i]))
		}
		if b, err := json.Marshal(cachePayload); err == nil {
			client.Set(ctx, modelListCacheKey, b, cacheTTLHour)
		}
	}
	return snapshot, nil
}

// modelCacheDTO 模型列表共享缓存（ai:model:list）传输对象：键名一律 snake_case，与 python
// （AiModelResult.model_dump(mode="json")）和 java（AiJsonUtils SNAKE_MAPPER）互认；
// 与对外 camelCase 的 vo.AiModelVO 严格隔离，改对外契约不得影响缓存格式（反之亦然）。
//
// create_time 为 go 侧补充字段（python 缓存不含），对端读到未知键会忽略。
type modelCacheDTO struct {
	ID                    int64           `json:"id"`
	ProviderID            int64           `json:"provider_id"`
	ModelID               string          `json:"model_id"`
	ModelType             string          `json:"model_type"`
	Dimension             *int64          `json:"dimension"`
	DisplayName           string          `json:"display_name"`
	MaxContextTokens      int             `json:"max_context_tokens"`
	MaxOutputTokens       int             `json:"max_output_tokens"`
	SupportsMultimodal    int8            `json:"supports_multimodal"`
	SupportsToolCall      int8            `json:"supports_tool_call"`
	SupportsStreaming     int8            `json:"supports_streaming"`
	SupportsPromptCache   int8            `json:"supports_prompt_cache"`
	SupportsStructuredOut int8            `json:"supports_structured_output"`
	ExtraRequestParams    json.RawMessage `json:"extra_request_params"`
	FallbackModelID       *int64          `json:"fallback_model_id"`
	PromptCachePrefixLen  int             `json:"prompt_cache_prefix_len"`
	Status                int8            `json:"status"`
	VipLevel              int8            `json:"vip_level"`
	LastTestStatus        int8            `json:"last_test_status"`
	LastTestAt            *time.Time      `json:"last_test_at"`
	LastTestError         *string         `json:"last_test_error"`
	Calls24h              *int64          `json:"calls_24h"`
	SuccessRate24h        *int            `json:"success_rate_24h"`
	LastCallAt            *time.Time      `json:"last_call_at"`
	SpeedTier             string          `json:"speed_tier"`
	IsFallbackTarget      bool            `json:"is_fallback_target"`
	CreateTime            *time.Time      `json:"create_time,omitempty"`
}

func modelToCache(item *vo.AiModelVO) modelCacheDTO {
	// create_time 为 go 侧补充字段：对端缓存不含时保持缺失，不回写零值时间
	var createTime *time.Time
	if !item.CreateTime.IsZero() {
		value := item.CreateTime
		createTime = &value
	}
	return modelCacheDTO{
		ID: item.ID, ProviderID: item.ProviderID, ModelID: item.ModelID, ModelType: item.ModelType,
		Dimension: item.Dimension, DisplayName: item.DisplayName,
		MaxContextTokens: item.MaxContextTokens, MaxOutputTokens: item.MaxOutputTokens,
		SupportsMultimodal: item.SupportsMultimodal, SupportsToolCall: item.SupportsToolCall,
		SupportsStreaming: item.SupportsStreaming, SupportsPromptCache: item.SupportsPromptCache,
		SupportsStructuredOut: item.SupportsStructuredOut,
		ExtraRequestParams:    item.ExtraRequestParams, FallbackModelID: item.FallbackModelID,
		PromptCachePrefixLen: item.PromptCachePrefixLen, Status: item.Status, VipLevel: item.VipLevel,
		LastTestStatus: item.LastTestStatus, LastTestAt: item.LastTestAt,
		LastTestError: item.LastTestError, Calls24h: item.Calls24h,
		SuccessRate24h: item.SuccessRate24h, LastCallAt: item.LastCallAt,
		SpeedTier: item.SpeedTier, IsFallbackTarget: item.IsFallbackTarget,
		CreateTime: createTime,
	}
}

func modelFromCache(dto *modelCacheDTO) vo.AiModelVO {
	item := vo.AiModelVO{
		ID: dto.ID, ProviderID: dto.ProviderID, ModelID: dto.ModelID, ModelType: dto.ModelType,
		Dimension: dto.Dimension, DisplayName: dto.DisplayName,
		MaxContextTokens: dto.MaxContextTokens, MaxOutputTokens: dto.MaxOutputTokens,
		SupportsMultimodal: dto.SupportsMultimodal, SupportsToolCall: dto.SupportsToolCall,
		SupportsStreaming: dto.SupportsStreaming, SupportsPromptCache: dto.SupportsPromptCache,
		SupportsStructuredOut: dto.SupportsStructuredOut,
		ExtraRequestParams:    dto.ExtraRequestParams, FallbackModelID: dto.FallbackModelID,
		PromptCachePrefixLen: dto.PromptCachePrefixLen, Status: dto.Status, VipLevel: dto.VipLevel,
		LastTestStatus: dto.LastTestStatus, LastTestAt: dto.LastTestAt,
		LastTestError: dto.LastTestError, Calls24h: dto.Calls24h,
		SuccessRate24h: dto.SuccessRate24h, LastCallAt: dto.LastCallAt,
		SpeedTier: dto.SpeedTier, IsFallbackTarget: dto.IsFallbackTarget,
	}
	if dto.CreateTime != nil {
		item.CreateTime = *dto.CreateTime
	}
	return item
}

// ValidateModelType 模型类型合法集校验（python `AiModelCreate/Update.model_type` 与
// `AiModelPageQuery.model_type` 均为 Literal[chat/embedding/rerank]）：创建/更新/列表筛选三处共用同一信息源。
// python 由 pydantic 在请求校验阶段报 A0400，go 无该层，在 service 层拦成同码错误。
func ValidateModelType(modelType string) error {
	switch modelType {
	case "chat", "embedding", "rerank":
		return nil
	}
	return common.NewBizError(common.PARAM_ERROR, "模型类型仅支持 chat/embedding/rerank")
}

func (s *ModelService) CreateModel(ctx context.Context, form *bo.AiModelCreateForm, operatorID int64) (*vo.AiModelVO, error) {
	modelType := form.ModelType
	if modelType == "" {
		modelType = "chat"
	}
	if err := ValidateModelType(modelType); err != nil {
		return nil, err
	}
	if modelType == "embedding" && form.Dimension == nil {
		return nil, common.NewBizError(common.PARAM_ERROR, "embedding 模型必须填写向量维度 dimension")
	}
	existing, err := s.modelRepo.GetByModelAndProvider(ctx, form.ModelID, form.ProviderID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "校验模型唯一性失败", err)
	}
	if existing != nil {
		return nil, common.NewBizError(common.DATA_EXISTS, "该模型+供应商组合已存在")
	}

	m := &model.SysAiModel{
		ProviderID:           form.ProviderID,
		ModelID:              form.ModelID,
		ModelType:            modelType,
		DisplayName:          form.DisplayName,
		MaxContextTokens:     4096,
		MaxOutputTokens:      4096,
		SupportsStreaming:    1,
		Status:               1,
		ExtraRequestParams:   form.ExtraRequestParams,
		FallbackModelID:      form.FallbackModelID,
		PromptCachePrefixLen: intOr(form.PromptCachePrefixLen, 0),
		VipLevel:             int8Or(form.VipLevel, 0),
	}
	// dimension 仅对 embedding 有意义，其他类型强制置空，避免残留脏数据
	if modelType == "embedding" {
		m.Dimension = form.Dimension
	}
	m.MaxContextTokens = intOr(form.MaxContextTokens, 4096)
	m.MaxOutputTokens = intOr(form.MaxOutputTokens, 4096)
	m.SupportsMultimodal = boolToInt8(form.SupportsMultimodal, false)
	m.SupportsToolCall = boolToInt8(form.SupportsToolCall, false)
	m.SupportsStreaming = boolToInt8(form.SupportsStreaming, true)
	m.SupportsPromptCache = boolToInt8(form.SupportsPromptCache, false)
	m.SupportsStructuredOut = boolToInt8(form.SupportsStructuredOut, false)
	m.Status = int8Or(form.Status, 1)
	m.CreateBy = operatorID
	m.UpdateBy = operatorID

	if err := s.modelRepo.Create(ctx, m); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建模型失败", err)
	}
	clearModelCache(ctx)
	result := toAiModelVO(m)
	return &result, nil
}

// UpdateModel 更新模型（modelType/dimension 创建后不可改；禁用触发下线通知）
func (s *ModelService) UpdateModel(ctx context.Context, modelID string, form *bo.AiModelUpdateForm, operatorID int64) (*vo.AiModelVO, error) {
	// 与 CreateModel 同口径：非法 model_type 先报 A0400（python pydantic 先于 service 拒绝），
	// 合法值才落到下面的"创建后不可修改"业务拒绝(A0500)，避免非法字面量被误报成不可修改
	if form.ModelType != nil {
		if err := ValidateModelType(*form.ModelType); err != nil {
			return nil, err
		}
	}
	m, err := s.modelRepo.GetByModelID(ctx, modelID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询模型失败", err)
	}
	if m == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "模型不存在")
	}
	if form.ModelType != nil || form.Dimension != nil {
		immutable := make([]string, 0, 2)
		if form.ModelType != nil {
			immutable = append(immutable, "model_type")
		}
		if form.Dimension != nil {
			immutable = append(immutable, "dimension")
		}
		sort.Strings(immutable)
		return nil, common.NewBizError(common.DATA_STATE_NOT_ALLOW,
			fmt.Sprintf("模型类型与向量维度创建后不可修改: %v", immutable))
	}

	updates := map[string]interface{}{"update_by": operatorID}
	if form.ProviderID != nil {
		updates["provider_id"] = *form.ProviderID
	}
	if form.DisplayName != nil {
		updates["display_name"] = *form.DisplayName
	}
	if form.MaxContextTokens != nil {
		updates["max_context_tokens"] = *form.MaxContextTokens
	}
	if form.MaxOutputTokens != nil {
		updates["max_output_tokens"] = *form.MaxOutputTokens
	}
	if form.SupportsMultimodal != nil {
		updates["supports_multimodal"] = boolToInt8(form.SupportsMultimodal, false)
	}
	if form.SupportsToolCall != nil {
		updates["supports_tool_call"] = boolToInt8(form.SupportsToolCall, false)
	}
	if form.SupportsStreaming != nil {
		updates["supports_streaming"] = boolToInt8(form.SupportsStreaming, false)
	}
	if form.SupportsPromptCache != nil {
		updates["supports_prompt_cache"] = boolToInt8(form.SupportsPromptCache, false)
	}
	if form.SupportsStructuredOut != nil {
		updates["supports_structured_output"] = boolToInt8(form.SupportsStructuredOut, false)
	}
	if provided, isNull := rawProvided(form.ExtraRequestParams); provided {
		if isNull {
			updates["extra_request_params"] = nil
		} else {
			updates["extra_request_params"] = form.ExtraRequestParams
		}
	}
	if provided, isNull := rawProvided(form.FallbackModelID); provided {
		if isNull {
			updates["fallback_model_id"] = nil
		} else {
			value, convErr := strconv.ParseInt(string(bytes.TrimSpace(form.FallbackModelID)), 10, 64)
			if convErr != nil {
				return nil, common.NewBizError(common.PARAM_ERROR, "fallbackModelId 格式不正确")
			}
			updates["fallback_model_id"] = value
		}
	}
	if form.PromptCachePrefixLen != nil {
		updates["prompt_cache_prefix_len"] = *form.PromptCachePrefixLen
	}
	if form.VipLevel != nil {
		updates["vip_level"] = *form.VipLevel
	}
	oldStatus := m.Status
	if form.Status != nil {
		updates["status"] = *form.Status
	}

	if err := s.modelRepo.Update(ctx, m.ID, updates); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "更新模型失败", err)
	}
	clearModelCache(ctx)

	updated, err := s.modelRepo.GetByPK(ctx, m.ID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询模型失败", err)
	}
	if updated == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "模型不存在")
	}
	// 禁用模型（status 1→0）即"标记即将下线"，向使用中会话推送替换模型推荐
	if oldStatus == 1 && updated.Status == 0 {
		s.notifyModelReplacement(ctx, updated)
	}
	result := toAiModelVO(updated)
	return &result, nil
}

func (s *ModelService) notifyModelReplacement(ctx context.Context, m *model.SysAiModel) {
	if s.notifier == nil {
		return
	}
	userIDs, err := s.modelRepo.ListActiveConversationUsers(ctx, m.ModelID)
	if err != nil || len(userIDs) == 0 {
		return
	}
	title := fmt.Sprintf("模型 %s 即将不可用", m.DisplayName)
	content := fmt.Sprintf("您正在使用的模型「%s」即将停用，暂未配置替代模型，请及时更换其他可用模型。", m.DisplayName)
	if m.FallbackModelID != nil {
		if fallback, fbErr := s.modelRepo.GetByPK(ctx, *m.FallbackModelID); fbErr == nil && fallback != nil && fallback.Status == 1 {
			content = fmt.Sprintf("您正在使用的模型「%s」即将停用，建议切换到替代模型「%s」。", m.DisplayName, fallback.DisplayName)
		}
	}
	// 通知失败不阻断模型更新（与 python 一致）
	_, _ = s.notifier.Send(ctx, &bo.MessageSendForm{
		Type:         "business",
		Title:        title,
		Content:      content,
		Priority:     3,
		RecipientIDs: userIDs,
		BizModule:    "ai_model",
		BizID:        m.ModelID,
	})
}

// DeleteModel 删除模型（活跃会话或降级链引用时拒绝）
func (s *ModelService) DeleteModel(ctx context.Context, modelID string, operatorID int64) error {
	m, err := s.modelRepo.GetByModelID(ctx, modelID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询模型失败", err)
	}
	if m == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "模型不存在")
	}
	active, err := s.modelRepo.CountActiveConversations(ctx, modelID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "统计活跃会话失败", err)
	}
	if active > 0 {
		return common.NewBizError(common.DATA_BIND_EXISTS, "存在活跃会话正在使用该模型，请先禁用（status=0）")
	}
	fallbackRefs, err := s.modelRepo.CountFallbackTargets(ctx, m.ID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "统计降级引用失败", err)
	}
	if fallbackRefs > 0 {
		return common.NewBizError(common.DATA_BIND_EXISTS, "存在启用模型的降级链引用该模型，请先调整其 fallback_model_id")
	}
	if err := s.modelRepo.SoftDeleteByIDs(ctx, []int64{m.ID}, operatorID); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除模型失败", err)
	}
	clearModelCache(ctx)
	return nil
}

// ==================== 用户售价（价格版本化） ====================

func (s *ModelService) ListModelPrices(ctx context.Context, modelID string, q *bo.ModelPriceQuery) (*vo.PageResult[vo.ModelPriceVO], error) {
	page, size := q.Page, q.Size
	prices, total, err := s.priceRepo.Paginate(ctx, page, size, modelID, q.ProviderID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询用户售价失败", err)
	}
	ids := make([]int64, 0, len(prices))
	for i := range prices {
		ids = append(ids, prices[i].ID)
	}
	details, err := s.priceRepo.ListDetails(ctx, ids)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询价格档位失败", err)
	}
	grouped := make(map[int64][]vo.ModelPriceDetailVO, len(ids))
	for i := range details {
		grouped[details[i].PriceID] = append(grouped[details[i].PriceID], vo.ModelPriceDetailVO{
			ID:        details[i].ID,
			PriceID:   details[i].PriceID,
			TokenType: details[i].TokenType,
			TimeSlot:  details[i].TimeSlot,
			MinTokens: details[i].MinTokens,
			MaxTokens: details[i].MaxTokens,
			UnitPrice: formatDecimal(details[i].UnitPrice),
		})
	}
	items := make([]vo.ModelPriceVO, 0, len(prices))
	for i := range prices {
		items = append(items, toModelPriceVO(&prices[i], grouped[prices[i].ID]))
	}
	return &vo.PageResult[vo.ModelPriceVO]{List: items, Total: total}, nil
}

func (s *ModelService) CreateModelPrice(ctx context.Context, modelID string, form *bo.ModelPriceCreateForm, operatorID int64) (*vo.ModelPriceVO, error) {
	unit := form.Unit
	if unit == "" {
		unit = "credits_per_million"
	}
	version, err := s.priceRepo.NextPriceVersion(ctx, modelID, form.ProviderID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询价格版本失败", err)
	}
	effectiveFrom := time.Now()
	if form.EffectiveFrom != nil {
		effectiveFrom = *form.EffectiveFrom
	}
	price := &model.SysAiModelPrice{
		ModelID:       modelID,
		ProviderID:    form.ProviderID,
		PriceVersion:  version,
		Unit:          unit,
		EffectiveFrom: effectiveFrom,
		EffectiveTo:   form.EffectiveTo,
		Status:        int8Or(form.Status, 1),
	}
	price.CreateBy = operatorID
	price.UpdateBy = operatorID
	if err := s.priceRepo.Create(ctx, price); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建用户售价失败", err)
	}

	details := make([]model.SysAiModelPriceDetail, 0, len(form.Details))
	for _, d := range form.Details {
		details = append(details, model.SysAiModelPriceDetail{
			PriceID:   price.ID,
			TokenType: d.TokenType,
			TimeSlot:  d.TimeSlot,
			MinTokens: d.MinTokens,
			MaxTokens: d.MaxTokens,
			UnitPrice: d.UnitPrice,
		})
	}
	for i := range details {
		details[i].CreateBy = operatorID
		details[i].UpdateBy = operatorID
	}
	if err := s.priceRepo.CreateDetails(ctx, details); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建价格档位失败", err)
	}
	result := toModelPriceVO(price, toDetailVOs(details))
	return &result, nil
}

func (s *ModelService) UpdateModelPrice(ctx context.Context, priceID int64, form *bo.ModelPriceUpdateForm, operatorID int64) (*vo.ModelPriceVO, error) {
	price, err := s.priceRepo.GetByID(ctx, priceID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询用户售价失败", err)
	}
	if price == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "用户售价不存在")
	}
	updates := map[string]interface{}{"update_by": operatorID}
	if form.Unit != nil {
		updates["unit"] = *form.Unit
	}
	if form.EffectiveFrom != nil {
		updates["effective_from"] = *form.EffectiveFrom
	}
	if form.EffectiveTo != nil {
		updates["effective_to"] = *form.EffectiveTo
	}
	if form.Status != nil {
		updates["status"] = *form.Status
	}
	if len(updates) > 1 {
		if err := s.priceRepo.Update(ctx, price.ID, updates); err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "更新用户售价失败", err)
		}
	}
	updated, err := s.priceRepo.GetByID(ctx, price.ID)
	if err != nil || updated == nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询用户售价失败", err)
	}
	details, err := s.priceRepo.ListDetails(ctx, []int64{updated.ID})
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询价格档位失败", err)
	}
	result := toModelPriceVO(updated, toDetailVOs(details))
	return &result, nil
}

func (s *ModelService) DeleteModelPrice(ctx context.Context, priceID int64) error {
	price, err := s.priceRepo.GetByID(ctx, priceID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询用户售价失败", err)
	}
	if price == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "用户售价不存在")
	}
	if err := s.priceRepo.SoftDelete(ctx, price.ID); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除用户售价失败", err)
	}
	return nil
}

// ==================== 内部方法 ====================

type modelUsageStat struct {
	Calls24h       *int64
	SuccessRate24h *int
	LastCallAt     *time.Time
}

// usageStats24h chat 按 llm_call 聚合（含失败），embedding/rerank 按计费流水聚合（成功才落账）
func (s *ModelService) usageStats24h(ctx context.Context, models []model.SysAiModel) (map[int64]modelUsageStat, error) {
	stats := make(map[int64]modelUsageStat)
	if len(models) == 0 {
		return stats, nil
	}
	since := time.Now().Add(-24 * time.Hour)
	chatModelIDs := make([]string, 0, len(models))
	chatPKByModelID := make(map[string]int64, len(models))
	kbModelIDs := make([]string, 0, len(models))
	kbPKByModelID := make(map[string]int64, len(models))
	for i := range models {
		switch models[i].ModelType {
		case "chat":
			chatModelIDs = append(chatModelIDs, models[i].ModelID)
			chatPKByModelID[models[i].ModelID] = models[i].ID
		case "embedding", "rerank":
			kbModelIDs = append(kbModelIDs, models[i].ModelID)
			kbPKByModelID[models[i].ModelID] = models[i].ID
		}
	}

	if len(chatModelIDs) > 0 {
		var rows []struct {
			Model  string     `gorm:"column:model"`
			Total  int64      `gorm:"column:total"`
			Ok     int64      `gorm:"column:ok"`
			LastAt *time.Time `gorm:"column:last_at"`
		}
		err := s.db.WithContext(ctx).Table("sys_ai_llm_call").
			Select("model, COUNT(*) AS total, SUM(CASE WHEN status = 1 THEN 1 ELSE 0 END) AS ok, MAX(create_time) AS last_at").
			Where("create_time >= ? AND model IN ?", since, chatModelIDs).
			Group("model").Scan(&rows).Error
		if err != nil {
			return nil, err
		}
		for _, row := range rows {
			pk, ok := chatPKByModelID[row.Model]
			if !ok {
				continue
			}
			stat := modelUsageStat{Calls24h: &row.Total, LastCallAt: row.LastAt}
			if row.Total > 0 {
				rate := int((row.Ok*100 + row.Total/2) / row.Total)
				stat.SuccessRate24h = &rate
			}
			stats[pk] = stat
		}
	}

	if len(kbModelIDs) > 0 {
		var rows []struct {
			Model  string     `gorm:"column:model"`
			Total  int64      `gorm:"column:total"`
			LastAt *time.Time `gorm:"column:last_at"`
		}
		err := s.db.WithContext(ctx).Table("sys_ai_billing").
			Select("model, COUNT(*) AS total, MAX(create_time) AS last_at").
			Where("create_time >= ? AND bill_type IN ? AND model IN ?", since, []string{"embedding", "rerank"}, kbModelIDs).
			Group("model").Scan(&rows).Error
		if err != nil {
			return nil, err
		}
		for _, row := range rows {
			pk, ok := kbPKByModelID[row.Model]
			if !ok {
				continue
			}
			stat := stats[pk]
			total := row.Total
			if stat.Calls24h != nil {
				total += *stat.Calls24h
			}
			stat.Calls24h = &total
			if row.LastAt != nil && (stat.LastCallAt == nil || row.LastAt.After(*stat.LastCallAt)) {
				stat.LastCallAt = row.LastAt
			}
			if total > 0 {
				rate := 100
				stat.SuccessRate24h = &rate
			}
			stats[pk] = stat
		}
	}
	return stats, nil
}

func toAiModelVO(m *model.SysAiModel) vo.AiModelVO {
	return vo.AiModelVO{
		ID:                    m.ID,
		ProviderID:            m.ProviderID,
		ModelID:               m.ModelID,
		ModelType:             m.ModelType,
		Dimension:             m.Dimension,
		DisplayName:           m.DisplayName,
		MaxContextTokens:      m.MaxContextTokens,
		MaxOutputTokens:       m.MaxOutputTokens,
		SupportsMultimodal:    m.SupportsMultimodal,
		SupportsToolCall:      m.SupportsToolCall,
		SupportsStreaming:     m.SupportsStreaming,
		SupportsPromptCache:   m.SupportsPromptCache,
		SupportsStructuredOut: m.SupportsStructuredOut,
		ExtraRequestParams:    m.ExtraRequestParams,
		FallbackModelID:       m.FallbackModelID,
		PromptCachePrefixLen:  m.PromptCachePrefixLen,
		Status:                m.Status,
		VipLevel:              m.VipLevel,
		LastTestStatus:        m.LastTestStatus,
		LastTestAt:            m.LastTestAt,
		LastTestError:         m.LastTestError,
		SpeedTier:             "unknown",
		CreateTime:            m.CreatedAt,
	}
}

func toModelPriceVO(p *model.SysAiModelPrice, details []vo.ModelPriceDetailVO) vo.ModelPriceVO {
	if details == nil {
		details = []vo.ModelPriceDetailVO{}
	}
	return vo.ModelPriceVO{
		ID:            p.ID,
		ModelID:       p.ModelID,
		ProviderID:    p.ProviderID,
		PriceVersion:  p.PriceVersion,
		Unit:          p.Unit,
		EffectiveFrom: p.EffectiveFrom,
		EffectiveTo:   p.EffectiveTo,
		Status:        p.Status,
		Details:       details,
		CreateTime:    p.CreatedAt,
		UpdateTime:    p.UpdatedAt,
	}
}

func toDetailVOs(details []model.SysAiModelPriceDetail) []vo.ModelPriceDetailVO {
	items := make([]vo.ModelPriceDetailVO, 0, len(details))
	for i := range details {
		items = append(items, vo.ModelPriceDetailVO{
			ID:        details[i].ID,
			PriceID:   details[i].PriceID,
			TokenType: details[i].TokenType,
			TimeSlot:  details[i].TimeSlot,
			MinTokens: details[i].MinTokens,
			MaxTokens: details[i].MaxTokens,
			UnitPrice: formatDecimal(details[i].UnitPrice),
		})
	}
	return items
}

func speedTierOf(snapshot map[string]any) string {
	raw, ok := snapshot["p95_latency_ms"]
	if !ok || raw == nil {
		return "unknown"
	}
	p95 := 0
	switch v := raw.(type) {
	case float64:
		p95 = int(v)
	case int:
		p95 = v
	case json.Number:
		if n, err := v.Int64(); err == nil {
			p95 = int(n)
		}
	default:
		if n, err := strconv.Atoi(fmt.Sprintf("%v", v)); err == nil {
			p95 = n
		}
	}
	switch {
	case p95 < 3000:
		return "fast"
	case p95 < 8000:
		return "medium"
	default:
		return "slow"
	}
}

// formatDecimal 对齐 python Decimal 的字符串序列化（SDK 类型为 string）
func formatDecimal(v float64) string {
	return strconv.FormatFloat(v, 'f', -1, 64)
}

func rawProvided(raw json.RawMessage) (bool, bool) {
	if len(raw) == 0 {
		return false, false
	}
	return true, string(bytes.TrimSpace(raw)) == "null"
}

func intOr(v *int, fallback int) int {
	if v == nil {
		return fallback
	}
	return *v
}

func int8Or(v *int8, fallback int8) int8 {
	if v == nil {
		return fallback
	}
	return *v
}

func boolToInt8(v *bool, fallback bool) int8 {
	value := fallback
	if v != nil {
		value = *v
	}
	if value {
		return 1
	}
	return 0
}

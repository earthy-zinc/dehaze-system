package com.pei.clarity.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.clarity.common.exception.BusinessException;
import com.pei.clarity.common.result.ResultCode;
import com.pei.clarity.common.util.AiJsonUtils;
import com.pei.clarity.mapper.SysAiModelMapper;
import com.pei.clarity.mapper.SysAiModelPriceDetailMapper;
import com.pei.clarity.mapper.SysAiModelPriceMapper;
import com.pei.clarity.model.entity.SysAiModel;
import com.pei.clarity.model.entity.SysAiModelPrice;
import com.pei.clarity.model.entity.SysAiModelPriceDetail;
import com.pei.clarity.model.form.AiModelForm;
import com.pei.clarity.model.form.AiModelUpdateForm;
import com.pei.clarity.model.form.MessageSendForm;
import com.pei.clarity.model.form.ModelPriceForm;
import com.pei.clarity.model.form.ModelPriceUpdateForm;
import com.pei.clarity.model.query.AiModelPageQuery;
import com.pei.clarity.model.query.ModelPriceQuery;
import com.pei.clarity.model.vo.AiModelVO;
import com.pei.clarity.model.vo.ModelPriceVO;
import com.pei.clarity.security.util.SecurityUtils;
import com.pei.clarity.service.AiDefaultModelResolver;
import com.pei.clarity.service.AiModelService;
import com.pei.clarity.service.AiProviderHealthService;
import com.pei.clarity.service.AiUserLevelResolver;
import com.pei.clarity.service.FavoriteService;
import com.pei.clarity.service.MessageService;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Slf4j
@Service
@RequiredArgsConstructor
public class AiModelServiceImpl extends ServiceImpl<SysAiModelMapper, SysAiModel> implements AiModelService {

    /** 启用模型列表缓存：与 python 共用同一键，写操作须失效同键 */
    private static final String MODEL_LIST_CACHE_KEY = "ai:model:list";

    private static final long MODEL_LIST_CACHE_TTL = 3600L;
    private static final String CACHE_INVALIDATION_CHANNEL = "cache:invalidation";
    private static final String INSTANCE_ID = "clarity-java-" + UUID.randomUUID();

    /** 速度档位阈值（与 python settings.AI_SPEED_TIER_* 同源，属纯技术参数，不进配置中心） */
    private static final int SPEED_TIER_FAST_P95_MS = 2000;

    private static final int SPEED_TIER_MEDIUM_P95_MS = 8000;

    /**
     * python {@code AiModelType} 为 {@code Literal["chat","embedding","rerank",...]} 全模态白名单，
     * 非法字面量在 python 属请求校验阶段（A0400）。白名单取值校验保持 service 层单一信息源（不搬 @Pattern 到 DTO），
     * 否则拼错的类型会直接落库并污染模型类型筛选与目录展示。
     */
    private static final Set<String> MODEL_TYPES = Set.of(
            "chat", "embedding", "rerank", "image_gen", "image_edit", "video_gen", "asr", "tts", "music_gen", "3d_gen");

    private static final String MODEL_TYPE_HINT =
            "模型类型仅支持 chat/embedding/rerank/image_gen/image_edit/video_gen/asr/tts/music_gen/3d_gen";

    private final SysAiModelPriceMapper priceMapper;
    private final SysAiModelPriceDetailMapper priceDetailMapper;
    private final AiUserLevelResolver userLevelResolver;
    private final AiProviderHealthService providerHealthService;
    private final MessageService messageService;
    private final FavoriteService favoriteService;
    private final StringRedisTemplate redis;
    private final ObjectMapper objectMapper;
    private final AiDefaultModelResolver aiDefaultModelResolver;

    // ==================== 模型 ====================

    @Override
    @Transactional(readOnly = true)
    public Page<AiModelVO> listModels(AiModelPageQuery query) {
        // 列表筛选同为 Literal 约束：python 对非法筛选值报 A0400（而非"过滤后空列表"），
        // 空串在 python 亦被 Literal 拒绝，故此处按非 null 校验、不额外放过空串
        if (query.getModelType() != null && !MODEL_TYPES.contains(query.getModelType())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, MODEL_TYPE_HINT);
        }
        String keyword = query.getKeyword();
        LambdaQueryWrapper<SysAiModel> wrapper = new LambdaQueryWrapper<SysAiModel>()
                .eq(CharSequenceUtil.isNotBlank(query.getModelType()), SysAiModel::getModelType, query.getModelType())
                .and(
                        CharSequenceUtil.isNotBlank(keyword),
                        q -> q.like(SysAiModel::getDisplayName, keyword).or().like(SysAiModel::getModelId, keyword));
        List<String> taskTags = parseTaskTags(query.getTaskTags());
        if (!taskTags.isEmpty()) {
            // task_tags 为 NULL 时 JSON_OVERLAPS 结果为 NULL（不命中），与"无标签不参与筛选"一致
            wrapper.apply("JSON_OVERLAPS(task_tags, CAST({0} AS JSON))", writeJsonList(taskTags));
        }
        wrapper.orderByDesc(SysAiModel::getId);
        Page<SysAiModel> page = this.page(new Page<>(query.getPageNum(), query.getPageSize()), wrapper);

        Map<Long, Map<String, Object>> stats = usageStats24h(page.getRecords());
        Page<AiModelVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        List<AiModelVO> list = new ArrayList<>(page.getRecords().size());
        for (SysAiModel model : page.getRecords()) {
            AiModelVO vo = toVO(model);
            Map<String, Object> stat = stats.get(model.getId());
            if (stat != null) {
                vo.setCalls24h((Integer) stat.get("calls24h"));
                vo.setSuccessRate24h((Integer) stat.get("successRate24h"));
                vo.setLastCallAt((LocalDateTime) stat.get("lastCallAt"));
            }
            list.add(vo);
        }
        result.setRecords(list);
        return result;
    }

    @Override
    @Transactional(readOnly = true)
    public List<AiModelVO> listEnabledModels(String modelType, String taskTags) {
        List<AiModelVO> models = readEnabledCache();
        if (models == null) {
            models = buildEnabledSnapshot();
            redis.opsForValue()
                    .set(MODEL_LIST_CACHE_KEY, AiJsonUtils.write(models), MODEL_LIST_CACHE_TTL, TimeUnit.SECONDS);
        }
        int userLevel = userLevelResolver.levelOf(SecurityUtils.getUserId());
        List<String> wanted = parseTaskTags(taskTags);
        List<AiModelVO> result = new ArrayList<>();
        for (AiModelVO model : models) {
            int vipLevel = model.getVipLevel() == null ? 0 : model.getVipLevel();
            if (vipLevel > userLevel) {
                continue;
            }
            if (modelType != null && !modelType.equals(model.getModelType())) {
                continue;
            }
            if (!wanted.isEmpty() && !taskTagsHit(model.getTaskTags(), wanted)) {
                continue;
            }
            result.add(model);
        }
        markPlatformDefaults(result);
        return result;
    }

    /**
     * 回填"是否平台默认"标记（isDefault）：按 model_type 分键读字典
     * （{@code ai_model.default_model}（chat）/ {@code default_model.{modelType}}），
     * 标记语义为"该模型是否为其类型的平台默认模型"——同一 model_id 配多供应商时该模型各行都为
     * true。**读取时计算而非写入共享缓存快照**：字典改动即失效，避免缓存 TTL 内标记滞后。
     */
    private void markPlatformDefaults(List<AiModelVO> models) {
        Map<String, String> defaultByType = new HashMap<>();
        for (AiModelVO model : models) {
            String type = model.getModelType();
            if (!defaultByType.containsKey(type)) {
                defaultByType.put(type, aiDefaultModelResolver.resolvePlatformDefaultModelId(type));
            }
        }
        for (AiModelVO model : models) {
            model.setIsDefault(Objects.equals(defaultByType.get(model.getModelType()), model.getModelId()));
        }
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public AiModelVO createModel(AiModelForm form) {
        if (form.getModelType() == null || !MODEL_TYPES.contains(form.getModelType())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, MODEL_TYPE_HINT);
        }
        if ("embedding".equals(form.getModelType()) && form.getDimension() == null) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "embedding 模型必须填写向量维度 dimension");
        }
        // dimension 仅对 embedding 有意义，其他类型强制置空，避免残留脏数据
        Long dimension = "embedding".equals(form.getModelType()) ? form.getDimension() : null;
        Long duplicate = this.baseMapper.selectCount(new LambdaQueryWrapper<SysAiModel>()
                .eq(SysAiModel::getModelId, form.getModelId())
                .eq(SysAiModel::getProviderId, form.getProviderId()));
        if (duplicate != null && duplicate > 0) {
            throw new BusinessException(ResultCode.DATA_EXISTS, "该模型+供应商组合已存在");
        }

        SysAiModel model = new SysAiModel();
        model.setProviderId(form.getProviderId());
        model.setModelId(form.getModelId());
        model.setModelType(form.getModelType());
        model.setTaskTags(writeJsonList(form.getTaskTags()));
        model.setDimension(dimension);
        model.setDisplayName(form.getDisplayName());
        model.setMaxContextTokens(form.getMaxContextTokens());
        model.setMaxOutputTokens(form.getMaxOutputTokens());
        model.setSupportsMultimodal(bool(form.getSupportsMultimodal()));
        model.setSupportsToolCall(bool(form.getSupportsToolCall()));
        model.setSupportsStreaming(bool(form.getSupportsStreaming()));
        model.setSupportsPromptCache(bool(form.getSupportsPromptCache()));
        model.setSupportsStructuredOutput(bool(form.getSupportsStructuredOutput()));
        model.setExtraRequestParams(writeJson(form.getExtraRequestParams()));
        model.setFallbackModelId(form.getFallbackModelId());
        model.setPromptCachePrefixLen(form.getPromptCachePrefixLen());
        model.setStatus(form.getStatus());
        model.setVipLevel(form.getVipLevel());
        this.save(model);
        clearModelCache();
        return toVO(model);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public AiModelVO updateModel(String modelId, AiModelUpdateForm form) {
        // 与 createModel 同口径：非法 modelType 先报 A0400（python pydantic 先于 service 拒绝），
        // 合法值才落到下面的"创建后不可修改"业务拒绝，避免拼错的值被误报成不可修改
        if (form.getModelType() != null && !MODEL_TYPES.contains(form.getModelType())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, MODEL_TYPE_HINT);
        }
        SysAiModel model = getByModelId(modelId);
        if (model == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "模型不存在");
        }
        if (form.getModelType() != null || form.getDimension() != null) {
            List<String> immutable = new ArrayList<>();
            if (form.getModelType() != null) {
                immutable.add("modelType");
            }
            if (form.getDimension() != null) {
                immutable.add("dimension");
            }
            throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "模型类型与向量维度创建后不可修改: " + immutable);
        }

        Integer oldStatus = model.getStatus();
        if (form.getProviderId() != null) {
            model.setProviderId(form.getProviderId());
        }
        if (form.getDisplayName() != null) {
            model.setDisplayName(form.getDisplayName());
        }
        if (form.getTaskTags() != null) {
            model.setTaskTags(writeJsonList(form.getTaskTags()));
        }
        if (form.getMaxContextTokens() != null) {
            model.setMaxContextTokens(form.getMaxContextTokens());
        }
        if (form.getMaxOutputTokens() != null) {
            model.setMaxOutputTokens(form.getMaxOutputTokens());
        }
        if (form.getSupportsMultimodal() != null) {
            model.setSupportsMultimodal(bool(form.getSupportsMultimodal()));
        }
        if (form.getSupportsToolCall() != null) {
            model.setSupportsToolCall(bool(form.getSupportsToolCall()));
        }
        if (form.getSupportsStreaming() != null) {
            model.setSupportsStreaming(bool(form.getSupportsStreaming()));
        }
        if (form.getSupportsPromptCache() != null) {
            model.setSupportsPromptCache(bool(form.getSupportsPromptCache()));
        }
        if (form.getSupportsStructuredOutput() != null) {
            model.setSupportsStructuredOutput(bool(form.getSupportsStructuredOutput()));
        }
        if (form.getExtraRequestParams() != null) {
            model.setExtraRequestParams(writeJson(form.getExtraRequestParams()));
        }
        if (form.getFallbackModelId() != null) {
            model.setFallbackModelId(form.getFallbackModelId());
        }
        if (form.getPromptCachePrefixLen() != null) {
            model.setPromptCachePrefixLen(form.getPromptCachePrefixLen());
        }
        if (form.getStatus() != null) {
            model.setStatus(form.getStatus());
        }
        if (form.getVipLevel() != null) {
            model.setVipLevel(form.getVipLevel());
        }
        this.updateById(model);

        // 禁用模型（status 1→0）即"标记即将下线"，向使用中会话推送替换模型推荐
        boolean disabling = form.getStatus() != null && oldStatus != null && oldStatus == 1 && form.getStatus() == 0;
        if (disabling) {
            notifyModelReplacement(model);
        }
        clearModelCache();
        return toVO(model);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deleteModel(String modelId) {
        SysAiModel model = getByModelId(modelId);
        if (model == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "模型不存在");
        }
        if (this.baseMapper.countActiveConversations(modelId) > 0) {
            throw new BusinessException(ResultCode.DATA_REFERENCED, "存在活跃会话正在使用该模型，请先禁用（status=0）");
        }
        // 被其他启用模型作为降级目标引用时不可删除，避免其降级链静默断裂
        Long fallbackRefs = this.baseMapper.selectCount(new LambdaQueryWrapper<SysAiModel>()
                .eq(SysAiModel::getFallbackModelId, model.getId())
                .eq(SysAiModel::getStatus, 1));
        if (fallbackRefs != null && fallbackRefs > 0) {
            throw new BusinessException(ResultCode.DATA_REFERENCED, "存在启用模型的降级链引用该模型，请先调整其 fallback_model_id");
        }
        this.removeById(model.getId());
        // 收藏失效联动：模型删除后其收藏标记失效（target_id 存 sys_ai_model.id，与收藏写入口径一致；
        // 与 python delete_model、数据集删除路径同序：软删 → 标记失效 → 清缓存）
        favoriteService.markInvalid("model", List.of(model.getId()));
        clearModelCache();
    }

    // ==================== 用户售价版本 ====================

    @Override
    @Transactional(readOnly = true)
    public Page<ModelPriceVO> listPrices(ModelPriceQuery query) {
        LambdaQueryWrapper<SysAiModelPrice> wrapper = new LambdaQueryWrapper<SysAiModelPrice>()
                .eq(CharSequenceUtil.isNotBlank(query.getModelId()), SysAiModelPrice::getModelId, query.getModelId())
                .eq(query.getProviderId() != null, SysAiModelPrice::getProviderId, query.getProviderId())
                .orderByDesc(SysAiModelPrice::getCreateTime)
                .orderByDesc(SysAiModelPrice::getId);
        Page<SysAiModelPrice> page = priceMapper.selectPage(new Page<>(query.getPage(), query.getSize()), wrapper);

        Page<ModelPriceVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        List<ModelPriceVO> records = new ArrayList<>(page.getRecords().size());
        for (SysAiModelPrice price : page.getRecords()) {
            records.add(toPriceVO(price, listDetails(price.getId())));
        }
        result.setRecords(records);
        return result;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public ModelPriceVO createPrice(ModelPriceForm form) {
        // billing_unit/usage_kind 未显式传时按模型类型派生：图像按张、asr 按秒、tts 按万字符、视频按帧，其余按百万 token
        String modelType = modelTypeOf(form.getModelId(), form.getProviderId());
        String defaultUsageKind = isUnitBilling(modelType) ? "unit" : "token";
        SysAiModelPrice price = new SysAiModelPrice();
        price.setModelId(form.getModelId());
        price.setProviderId(form.getProviderId());
        price.setPriceVersion(priceMapper.nextPriceVersion(form.getModelId(), form.getProviderId()));
        price.setUnit(form.getUnit());
        price.setBillingUnit(form.getBillingUnit() != null ? form.getBillingUnit() : defaultBillingUnit(modelType));
        price.setEffectiveFrom(form.getEffectiveFrom() != null ? form.getEffectiveFrom() : LocalDateTime.now());
        price.setEffectiveTo(form.getEffectiveTo());
        price.setStatus(form.getStatus());
        priceMapper.insert(price);

        List<SysAiModelPriceDetail> details = new ArrayList<>();
        for (ModelPriceForm.Detail detail : form.getDetails()) {
            SysAiModelPriceDetail entity = new SysAiModelPriceDetail();
            entity.setPriceId(price.getId());
            entity.setTokenType(detail.getTokenType());
            entity.setUsageKind(detail.getUsageKind() != null ? detail.getUsageKind() : defaultUsageKind);
            entity.setTimeSlot(detail.getTimeSlot());
            entity.setMinTokens(detail.getMinTokens());
            entity.setMaxTokens(detail.getMaxTokens());
            entity.setUnitPrice(detail.getUnitPrice());
            priceDetailMapper.insert(entity);
            details.add(entity);
        }
        return toPriceVO(price, details);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public ModelPriceVO updatePrice(Long priceId, ModelPriceUpdateForm form) {
        SysAiModelPrice price = priceMapper.selectById(priceId);
        if (price == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "用户售价不存在");
        }
        if (form.getUnit() != null) {
            price.setUnit(form.getUnit());
        }
        if (form.getBillingUnit() != null) {
            price.setBillingUnit(form.getBillingUnit());
        }
        if (form.getEffectiveFrom() != null) {
            price.setEffectiveFrom(form.getEffectiveFrom());
        }
        if (form.getEffectiveTo() != null) {
            price.setEffectiveTo(form.getEffectiveTo());
        }
        if (form.getStatus() != null) {
            price.setStatus(form.getStatus());
        }
        priceMapper.updateById(price);
        return toPriceVO(price, listDetails(priceId));
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deletePrice(Long priceId) {
        SysAiModelPrice price = priceMapper.selectById(priceId);
        if (price == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "用户售价不存在");
        }
        priceMapper.deleteById(priceId);
        priceDetailMapper.delete(
                new LambdaQueryWrapper<SysAiModelPriceDetail>().eq(SysAiModelPriceDetail::getPriceId, priceId));
    }

    // ==================== 内部实现 ====================

    /**
     * 按业务键 model_id 定位模型行。
     *
     * <p>同一 model_id 可配多供应商，取主键最小者（python 侧同键查询在单行时行为一致）。
     */
    private SysAiModel getByModelId(String modelId) {
        List<SysAiModel> rows = this.list(new LambdaQueryWrapper<SysAiModel>()
                .eq(SysAiModel::getModelId, modelId)
                .orderByAsc(SysAiModel::getId));
        return rows.isEmpty() ? null : rows.get(0);
    }

    private List<SysAiModelPriceDetail> listDetails(Long priceId) {
        return priceDetailMapper.selectList(new LambdaQueryWrapper<SysAiModelPriceDetail>()
                .eq(SysAiModelPriceDetail::getPriceId, priceId)
                .orderByAsc(SysAiModelPriceDetail::getId));
    }

    /** 价格归属模型的类型（价格按 model_id + provider_id 唯一定位模型），模型不存在返回 null */
    private String modelTypeOf(String modelId, Long providerId) {
        SysAiModel model = this.getBaseMapper()
                .selectOne(new LambdaQueryWrapper<SysAiModel>()
                        .eq(SysAiModel::getModelId, modelId)
                        .eq(SysAiModel::getProviderId, providerId));
        return model == null ? null : model.getModelType();
    }

    /** 未显式指定 billing_unit 时按模型类型派生计量单位 */
    private static String defaultBillingUnit(String modelType) {
        if (modelType == null) {
            return "million_tokens";
        }
        return switch (modelType) {
            case "image_gen", "image_edit" -> "image";
            case "asr" -> "second";
            case "tts" -> "ten_thousand_chars";
            case "video_gen" -> "frame";
            default -> "million_tokens";
        };
    }

    private static boolean isUnitBilling(String modelType) {
        return !"million_tokens".equals(defaultBillingUnit(modelType));
    }

    private ModelPriceVO toPriceVO(SysAiModelPrice price, List<SysAiModelPriceDetail> details) {
        ModelPriceVO vo = new ModelPriceVO();
        vo.setId(price.getId());
        vo.setModelId(price.getModelId());
        vo.setProviderId(price.getProviderId());
        vo.setPriceVersion(price.getPriceVersion());
        vo.setUnit(price.getUnit());
        vo.setBillingUnit(price.getBillingUnit());
        vo.setEffectiveFrom(price.getEffectiveFrom());
        vo.setEffectiveTo(price.getEffectiveTo());
        vo.setStatus(price.getStatus());
        vo.setCreateTime(price.getCreateTime());
        vo.setUpdateTime(price.getUpdateTime());
        List<ModelPriceVO.Detail> detailVOs = new ArrayList<>(details.size());
        for (SysAiModelPriceDetail detail : details) {
            ModelPriceVO.Detail detailVO = new ModelPriceVO.Detail();
            detailVO.setId(detail.getId());
            detailVO.setPriceId(detail.getPriceId());
            detailVO.setTokenType(detail.getTokenType());
            detailVO.setUsageKind(detail.getUsageKind());
            detailVO.setTimeSlot(detail.getTimeSlot());
            detailVO.setMinTokens(detail.getMinTokens());
            detailVO.setMaxTokens(detail.getMaxTokens());
            detailVO.setUnitPrice(detail.getUnitPrice());
            detailVOs.add(detailVO);
        }
        vo.setDetails(detailVOs);
        return vo;
    }

    private AiModelVO toVO(SysAiModel model) {
        AiModelVO vo = new AiModelVO();
        vo.setId(model.getId());
        vo.setProviderId(model.getProviderId());
        vo.setModelId(model.getModelId());
        vo.setModelType(model.getModelType());
        vo.setTaskTags(readJsonList(model.getTaskTags()));
        vo.setDimension(model.getDimension());
        vo.setDisplayName(model.getDisplayName());
        vo.setMaxContextTokens(model.getMaxContextTokens());
        vo.setMaxOutputTokens(model.getMaxOutputTokens());
        vo.setSupportsMultimodal(model.getSupportsMultimodal());
        vo.setSupportsToolCall(model.getSupportsToolCall());
        vo.setSupportsStreaming(model.getSupportsStreaming());
        vo.setSupportsPromptCache(model.getSupportsPromptCache());
        vo.setSupportsStructuredOutput(model.getSupportsStructuredOutput());
        vo.setExtraRequestParams(readJsonMap(model.getExtraRequestParams()));
        vo.setFallbackModelId(model.getFallbackModelId());
        vo.setPromptCachePrefixLen(model.getPromptCachePrefixLen());
        vo.setStatus(model.getStatus());
        vo.setVipLevel(model.getVipLevel());
        vo.setLastTestStatus(model.getLastTestStatus());
        vo.setLastTestAt(model.getLastTestAt());
        vo.setLastTestError(model.getLastTestError());
        vo.setCreateTime(model.getCreateTime());
        return vo;
    }

    /** 构建启用模型快照：降级标识 + 由供应商健康快照 P95 推导速度档位 */
    private List<AiModelVO> buildEnabledSnapshot() {
        List<SysAiModel> models = this.list(new LambdaQueryWrapper<SysAiModel>()
                .eq(SysAiModel::getStatus, 1)
                .orderByAsc(SysAiModel::getProviderId)
                .orderByAsc(SysAiModel::getDisplayName));
        Set<Long> fallbackTargets = new HashSet<>();
        for (SysAiModel model : models) {
            if (model.getFallbackModelId() != null) {
                fallbackTargets.add(model.getFallbackModelId());
            }
        }
        List<AiModelVO> snapshot = new ArrayList<>(models.size());
        for (SysAiModel model : models) {
            AiModelVO vo = toVO(model);
            vo.setSpeedTier(speedTierOf(model.getProviderId()));
            vo.setIsFallbackTarget(fallbackTargets.contains(model.getId()));
            snapshot.add(vo);
        }
        return snapshot;
    }

    /**
     * 读取启用模型缓存：缓存与 python 同键同格式（snake_case 快照），
     * 未命中或格式不可解析时回源数据库。
     */
    private List<AiModelVO> readEnabledCache() {
        String raw = redis.opsForValue().get(MODEL_LIST_CACHE_KEY);
        if (CharSequenceUtil.isBlank(raw)) {
            return null;
        }
        List<AiModelVO> cached = AiJsonUtils.read(raw, new TypeReference<List<AiModelVO>>() {});
        if (cached == null) {
            log.warn("启用模型缓存解析失败，删除后回源");
            redis.delete(MODEL_LIST_CACHE_KEY);
        }
        return cached;
    }

    /** 由供应商健康快照 P95 延迟推导速度档位（fast/medium/slow/unknown） */
    private String speedTierOf(Long providerId) {
        Object p95;
        try {
            p95 = providerHealthService.getSnapshot(providerId).get("p95_latency_ms");
        } catch (Exception e) {
            log.warn("读取供应商健康快照失败: providerId={} err={}", providerId, e.getMessage());
            return "unknown";
        }
        if (!(p95 instanceof Number number)) {
            return "unknown";
        }
        int latency = number.intValue();
        if (latency < SPEED_TIER_FAST_P95_MS) {
            return "fast";
        }
        if (latency < SPEED_TIER_MEDIUM_P95_MS) {
            return "medium";
        }
        return "slow";
    }

    /** 模型下线通知：向使用该模型的所有活跃会话用户推送替换模型推荐（站内信） */
    private void notifyModelReplacement(SysAiModel model) {
        List<Long> userIds = this.baseMapper.selectActiveConversationUserIds(model.getModelId());
        if (userIds == null || userIds.isEmpty()) {
            return;
        }
        SysAiModel fallback = model.getFallbackModelId() == null
                ? null
                : this.getOne(new LambdaQueryWrapper<SysAiModel>()
                        .eq(SysAiModel::getId, model.getFallbackModelId())
                        .eq(SysAiModel::getStatus, 1));
        String title = "模型 " + model.getDisplayName() + " 即将不可用";
        String content = fallback != null
                ? "您正在使用的模型「" + model.getDisplayName() + "」即将停用，建议切换到替代模型「" + fallback.getDisplayName() + "」。"
                : "您正在使用的模型「" + model.getDisplayName() + "」即将停用，暂未配置替代模型，请及时更换其他可用模型。";
        try {
            MessageSendForm form = new MessageSendForm();
            form.setType("business");
            form.setTitle(title);
            form.setContent(content);
            form.setPriority(3);
            form.setRecipientIds(userIds);
            form.setBizModule("ai_model");
            form.setBizId(model.getModelId());
            messageService.send(form);
        } catch (Exception e) {
            log.warn("模型下线通知失败: modelId={} err={}", model.getModelId(), e.getMessage());
        }
    }

    /** chat 按 llm_call 聚合、embedding/rerank 按计费流水聚合，返回 {pk: 统计} */
    private Map<Long, Map<String, Object>> usageStats24h(List<SysAiModel> models) {
        Map<Long, Map<String, Object>> stats = new HashMap<>();
        if (models.isEmpty()) {
            return stats;
        }
        LocalDateTime since = LocalDateTime.now().minusHours(24);
        Map<String, Long> chatPkByModelId = new HashMap<>();
        Map<String, Long> kbPkByModelId = new HashMap<>();
        for (SysAiModel model : models) {
            if ("chat".equals(model.getModelType())) {
                chatPkByModelId.put(model.getModelId(), model.getId());
            } else if ("embedding".equals(model.getModelType()) || "rerank".equals(model.getModelType())) {
                kbPkByModelId.put(model.getModelId(), model.getId());
            }
        }
        if (!chatPkByModelId.isEmpty()) {
            List<Map<String, Object>> rows =
                    this.baseMapper.selectChatUsage24h(since, new ArrayList<>(chatPkByModelId.keySet()));
            for (Map<String, Object> row : rows) {
                Long pk = chatPkByModelId.get(str(row.get("model")));
                if (pk == null) {
                    continue;
                }
                int total = intOf(row.get("total"));
                int ok = intOf(row.get("ok"));
                Map<String, Object> stat = new LinkedHashMap<>();
                stat.put("calls24h", total);
                stat.put("successRate24h", total > 0 ? Math.round(ok * 100f / total) : null);
                stat.put("lastCallAt", toLocalDateTime(row.get("lastAt")));
                stats.put(pk, stat);
            }
        }
        if (!kbPkByModelId.isEmpty()) {
            List<Map<String, Object>> rows =
                    this.baseMapper.selectKbUsage24h(since, new ArrayList<>(kbPkByModelId.keySet()));
            for (Map<String, Object> row : rows) {
                Long pk = kbPkByModelId.get(str(row.get("model")));
                if (pk == null) {
                    continue;
                }
                Map<String, Object> stat = stats.computeIfAbsent(pk, k -> {
                    Map<String, Object> created = new LinkedHashMap<>();
                    created.put("calls24h", 0);
                    created.put("successRate24h", null);
                    created.put("lastCallAt", null);
                    return created;
                });
                stat.put("calls24h", intOf(stat.get("calls24h")) + intOf(row.get("total")));
                LocalDateTime lastAt = toLocalDateTime(row.get("lastAt"));
                LocalDateTime existing = (LocalDateTime) stat.get("lastCallAt");
                if (lastAt != null && (existing == null || lastAt.isAfter(existing))) {
                    stat.put("lastCallAt", lastAt);
                }
            }
        }
        return stats;
    }

    /**
     * 失效启用模型缓存：删除 L2 同键并向 cache:invalidation 广播，
     * 使 python 各实例的 L1 进程内缓存同步失效（跨端一致）。
     */
    private void clearModelCache() {
        redis.delete(MODEL_LIST_CACHE_KEY);
        try {
            Map<String, Object> message = new LinkedHashMap<>();
            message.put("type", "key");
            message.put("key", MODEL_LIST_CACHE_KEY);
            message.put("senderId", INSTANCE_ID);
            redis.convertAndSend(CACHE_INVALIDATION_CHANNEL, objectMapper.writeValueAsString(message));
        } catch (Exception e) {
            log.warn("启用模型缓存失效广播失败: {}", e.getMessage());
        }
    }

    private Map<String, Object> readJsonMap(String json) {
        if (CharSequenceUtil.isBlank(json)) {
            return null;
        }
        try {
            return objectMapper.readValue(json, new TypeReference<Map<String, Object>>() {});
        } catch (Exception e) {
            log.warn("模型扩展参数解析失败: {}", e.getMessage());
            return null;
        }
    }

    private String writeJson(Map<String, Object> value) {
        if (value == null || value.isEmpty()) {
            return null;
        }
        try {
            return objectMapper.writeValueAsString(value);
        } catch (Exception e) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "厂商私有请求参数格式非法");
        }
    }

    /** task_tags 为 JSON 数组文本：null 不改；空数组写入 []（与 python 全量替换语义一致） */
    private List<String> readJsonList(String json) {
        if (CharSequenceUtil.isBlank(json)) {
            return null;
        }
        try {
            return objectMapper.readValue(json, new TypeReference<List<String>>() {});
        } catch (Exception e) {
            log.warn("模型任务标签解析失败: {}", e.getMessage());
            return null;
        }
    }

    private String writeJsonList(List<String> value) {
        if (value == null) {
            return null;
        }
        try {
            return objectMapper.writeValueAsString(value);
        } catch (Exception e) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "任务归类标签格式非法");
        }
    }

    /** taskTags 查询参数：逗号分隔，去空白与空项；空结果表示不筛选 */
    private static List<String> parseTaskTags(String raw) {
        if (CharSequenceUtil.isBlank(raw)) {
            return List.of();
        }
        List<String> tags = new ArrayList<>();
        for (String part : raw.split(",")) {
            String tag = part.trim();
            if (!tag.isEmpty()) {
                tags.add(tag);
            }
        }
        return tags;
    }

    /** task_tags 数组与筛选标签有交集即命中（无标签行不参与筛选） */
    private static boolean taskTagsHit(List<String> owned, List<String> wanted) {
        if (owned == null || owned.isEmpty()) {
            return false;
        }
        for (String tag : owned) {
            if (wanted.contains(tag)) {
                return true;
            }
        }
        return false;
    }

    private static int bool(Boolean value) {
        return Boolean.TRUE.equals(value) ? 1 : 0;
    }

    private static String str(Object value) {
        return value == null ? null : String.valueOf(value);
    }

    private static int intOf(Object value) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        if (value == null) {
            return 0;
        }
        try {
            return Integer.parseInt(String.valueOf(value));
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    /** 聚合 SQL 的 datetime 列在不同驱动配置下可能为 LocalDateTime 或 Timestamp，统一归一 */
    private static LocalDateTime toLocalDateTime(Object value) {
        if (value instanceof LocalDateTime localDateTime) {
            return localDateTime;
        }
        if (value instanceof java.sql.Timestamp timestamp) {
            return timestamp.toLocalDateTime();
        }
        return null;
    }
}

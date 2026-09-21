package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fasterxml.jackson.core.type.TypeReference;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.AiCredentialCipher;
import com.pei.dehaze.common.util.AiJsonUtils;
import com.pei.dehaze.mapper.SysAiModelMapper;
import com.pei.dehaze.mapper.SysAiProviderKeyMapper;
import com.pei.dehaze.mapper.SysAiProviderMapper;
import com.pei.dehaze.model.entity.SysAiModel;
import com.pei.dehaze.model.entity.SysAiProvider;
import com.pei.dehaze.model.entity.SysAiProviderKey;
import com.pei.dehaze.model.form.ProviderForm;
import com.pei.dehaze.model.form.ProviderKeyForm;
import com.pei.dehaze.model.form.ProviderKeyUpdateForm;
import com.pei.dehaze.model.form.ProviderUpdateForm;
import com.pei.dehaze.model.form.UserIdentityForwardForm;
import com.pei.dehaze.model.query.ProviderPageQuery;
import com.pei.dehaze.model.vo.ProviderEnabledVO;
import com.pei.dehaze.model.vo.ProviderKeyVO;
import com.pei.dehaze.model.vo.ProviderVO;
import com.pei.dehaze.service.AiProviderHealthService;
import com.pei.dehaze.service.AiProviderService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

@Slf4j
@Service
@RequiredArgsConstructor
public class AiProviderServiceImpl extends ServiceImpl<SysAiProviderMapper, SysAiProvider>
        implements AiProviderService {

    /** 启用供应商列表缓存：与 python 共用同一键，写操作须失效同键 */
    private static final String PROVIDER_LIST_CACHE_KEY = "ai:provider:list";
    private static final long PROVIDER_LIST_CACHE_TTL = 3600L;
    private static final String CACHE_INVALIDATION_CHANNEL = "cache:invalidation";
    private static final String INSTANCE_ID = "dehaze-java-" + UUID.randomUUID();

    private final SysAiProviderKeyMapper keyMapper;
    private final SysAiModelMapper modelMapper;
    private final AiProviderHealthService providerHealthService;
    private final AiCredentialCipher cipher;
    private final StringRedisTemplate redis;
    private final com.fasterxml.jackson.databind.ObjectMapper objectMapper;

    // ==================== 供应商 ====================

    @Override
    @Transactional(readOnly = true)
    public Page<ProviderVO> listProviders(ProviderPageQuery query) {
        String keyword = query.getKeyword();
        LambdaQueryWrapper<SysAiProvider> wrapper = new LambdaQueryWrapper<SysAiProvider>()
                .and(CharSequenceUtil.isNotBlank(keyword), q -> q
                        .like(SysAiProvider::getDisplayName, keyword)
                        .or()
                        .like(SysAiProvider::getProviderCode, keyword))
                .orderByAsc(SysAiProvider::getSortOrder)
                .orderByAsc(SysAiProvider::getId);
        Page<SysAiProvider> page = this.page(new Page<>(query.getPageNum(), query.getPageSize()), wrapper);

        Page<ProviderVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        List<ProviderVO> records = new ArrayList<>(page.getRecords().size());
        for (SysAiProvider provider : page.getRecords()) {
            ProviderVO vo = toVO(provider);
            vo.setHealth(providerHealthService.getStatus(provider.getId()));
            records.add(vo);
        }
        result.setRecords(records);
        return result;
    }

    @Override
    @Transactional(readOnly = true)
    public List<ProviderEnabledVO> listEnabledProviders() {
        String raw = redis.opsForValue().get(PROVIDER_LIST_CACHE_KEY);
        if (CharSequenceUtil.isNotBlank(raw)) {
            List<ProviderEnabledVO> cached = AiJsonUtils.read(raw, new TypeReference<List<ProviderEnabledVO>>() {
            });
            if (cached != null) {
                return cached;
            }
            log.warn("启用供应商缓存解析失败，删除后回源");
            redis.delete(PROVIDER_LIST_CACHE_KEY);
        }
        List<SysAiProvider> providers = this.list(new LambdaQueryWrapper<SysAiProvider>()
                .eq(SysAiProvider::getStatus, 1)
                .orderByAsc(SysAiProvider::getSortOrder)
                .orderByAsc(SysAiProvider::getId));
        List<ProviderEnabledVO> items = new ArrayList<>(providers.size());
        for (SysAiProvider provider : providers) {
            ProviderEnabledVO vo = new ProviderEnabledVO();
            vo.setId(provider.getId());
            vo.setProviderCode(provider.getProviderCode());
            vo.setDisplayName(provider.getDisplayName());
            vo.setProtocolType(provider.getProtocolType());
            vo.setStatus(provider.getStatus());
            items.add(vo);
        }
        redis.opsForValue().set(PROVIDER_LIST_CACHE_KEY, AiJsonUtils.write(items),
                PROVIDER_LIST_CACHE_TTL, TimeUnit.SECONDS);
        return items;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public ProviderVO createProvider(ProviderForm form) {
        if (this.baseMapper.countByProviderCodeIncludingDeleted(form.getProviderCode()) > 0) {
            Long active = this.baseMapper.selectCount(new LambdaQueryWrapper<SysAiProvider>()
                    .eq(SysAiProvider::getProviderCode, form.getProviderCode()));
            if (active != null && active > 0) {
                throw new BusinessException(ResultCode.DATA_EXISTS, "供应商编码已存在");
            }
            throw new BusinessException(ResultCode.DATA_EXISTS, "供应商编码已被历史记录占用，不可复用");
        }
        SysAiProvider provider = new SysAiProvider();
        provider.setProviderCode(form.getProviderCode());
        provider.setDisplayName(form.getDisplayName());
        provider.setApiBaseUrl(form.getApiBaseUrl());
        provider.setProtocolType(form.getProtocolType());
        provider.setAuthType(form.getAuthType());
        provider.setDefaultHeaders(AiJsonUtils.write(form.getDefaultHeaders()));
        provider.setSortOrder(form.getSortOrder());
        provider.setHealthCheckEnabled(form.getHealthCheckEnabled());
        provider.setUserIdentityForward(AiJsonUtils.write(form.getUserIdentityForward()));
        provider.setRemark(form.getRemark());
        provider.setStatus(form.getStatus());
        this.save(provider);
        clearProviderCache();
        providerHealthService.setHealthCheckEnabled(provider.getId(),
                provider.getHealthCheckEnabled() != null && provider.getHealthCheckEnabled() == 1);
        return toVO(provider);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public ProviderVO updateProvider(Long providerId, ProviderUpdateForm form) {
        SysAiProvider provider = getProviderOrRaise(providerId);
        if (form.getDisplayName() != null) {
            provider.setDisplayName(form.getDisplayName());
        }
        if (form.getApiBaseUrl() != null) {
            provider.setApiBaseUrl(form.getApiBaseUrl());
        }
        if (form.getProtocolType() != null) {
            provider.setProtocolType(form.getProtocolType());
        }
        if (form.getAuthType() != null) {
            provider.setAuthType(form.getAuthType());
        }
        if (form.getDefaultHeaders() != null) {
            provider.setDefaultHeaders(AiJsonUtils.write(form.getDefaultHeaders()));
        }
        if (form.getSortOrder() != null) {
            provider.setSortOrder(form.getSortOrder());
        }
        if (form.getHealthCheckEnabled() != null) {
            provider.setHealthCheckEnabled(form.getHealthCheckEnabled());
        }
        if (form.getUserIdentityForward() != null) {
            provider.setUserIdentityForward(AiJsonUtils.write(form.getUserIdentityForward()));
        }
        if (form.getRemark() != null) {
            provider.setRemark(form.getRemark());
        }
        if (form.getStatus() != null) {
            provider.setStatus(form.getStatus());
        }
        this.updateById(provider);
        clearProviderCache();
        providerHealthService.setHealthCheckEnabled(providerId,
                provider.getHealthCheckEnabled() != null && provider.getHealthCheckEnabled() == 1);
        return toVO(provider);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deleteProvider(Long providerId) {
        getProviderOrRaise(providerId);
        Long bound = modelMapper.selectCount(new LambdaQueryWrapper<SysAiModel>()
                .eq(SysAiModel::getProviderId, providerId));
        if (bound != null && bound > 0) {
            throw new BusinessException(ResultCode.DATA_BIND_EXISTS,
                    "存在模型引用该供应商（含禁用模型），请先删除或转移关联模型");
        }
        this.removeById(providerId);
        clearProviderCache();
        providerHealthService.clearProviderHealth(providerId);
    }

    // ==================== API Key ====================

    @Override
    @Transactional(readOnly = true)
    public List<ProviderKeyVO> listKeys(Long providerId) {
        getProviderOrRaise(providerId);
        List<SysAiProviderKey> keys = keyMapper.selectList(new LambdaQueryWrapper<SysAiProviderKey>()
                .eq(SysAiProviderKey::getProviderId, providerId)
                .orderByAsc(SysAiProviderKey::getPriority)
                .orderByAsc(SysAiProviderKey::getId));
        List<ProviderKeyVO> items = new ArrayList<>(keys.size());
        for (SysAiProviderKey key : keys) {
            items.add(toKeyVO(key));
        }
        return items;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public ProviderKeyVO createKey(Long providerId, ProviderKeyForm form) {
        getProviderOrRaise(providerId);
        String keyHash = cipher.hashKey(form.getKey());
        Long duplicate = keyMapper.selectCount(new LambdaQueryWrapper<SysAiProviderKey>()
                .eq(SysAiProviderKey::getKeyHash, keyHash));
        if (duplicate != null && duplicate > 0) {
            throw new BusinessException(ResultCode.DATA_EXISTS, "该 API Key 已存在");
        }
        SysAiProviderKey key = new SysAiProviderKey();
        key.setProviderId(providerId);
        key.setName(form.getName());
        key.setKeyHash(keyHash);
        key.setKeyPrefix(cipher.maskKey(form.getKey()));
        key.setKeyCipher(cipher.encrypt(form.getKey()));
        key.setStatus(form.getStatus());
        key.setPriority(form.getPriority());
        key.setWeight(form.getWeight());
        key.setDailyQuota(form.getDailyQuota());
        key.setRpmLimit(form.getRpmLimit());
        key.setExpiresAt(form.getExpiresAt());
        keyMapper.insert(key);
        return toKeyVO(key);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public ProviderKeyVO updateKey(Long providerId, Long keyId, ProviderKeyUpdateForm form) {
        SysAiProviderKey key = getKeyOrRaise(providerId, keyId);
        if (form.getName() != null) {
            key.setName(form.getName());
        }
        if (form.getPriority() != null) {
            key.setPriority(form.getPriority());
        }
        if (form.getWeight() != null) {
            key.setWeight(form.getWeight());
        }
        if (form.getStatus() != null) {
            key.setStatus(form.getStatus());
        }
        if (form.getDailyQuota() != null) {
            key.setDailyQuota(form.getDailyQuota());
        }
        if (form.getRpmLimit() != null) {
            key.setRpmLimit(form.getRpmLimit());
        }
        if (form.getExpiresAt() != null) {
            key.setExpiresAt(form.getExpiresAt());
        }
        keyMapper.updateById(key);
        return toKeyVO(key);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deleteKey(Long providerId, Long keyId) {
        SysAiProviderKey key = getKeyOrRaise(providerId, keyId);
        if (key.getStatus() != null && key.getStatus() == 1) {
            Long enabled = keyMapper.selectCount(new LambdaQueryWrapper<SysAiProviderKey>()
                    .eq(SysAiProviderKey::getProviderId, providerId)
                    .eq(SysAiProviderKey::getStatus, 1));
            if (enabled != null && enabled <= 1) {
                throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW,
                        "该供应商唯一启用 Key，不可删除，请先新增其他 Key 或禁用后再删除");
            }
        }
        keyMapper.deleteById(keyId);
    }

    @Override
    public void closeCircuit(Long providerId) {
        getProviderOrRaise(providerId);
        providerHealthService.closeCircuit(providerId);
    }

    // ==================== 内部实现 ====================

    private SysAiProvider getProviderOrRaise(Long providerId) {
        SysAiProvider provider = this.getById(providerId);
        if (provider == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "供应商不存在");
        }
        return provider;
    }

    private SysAiProviderKey getKeyOrRaise(Long providerId, Long keyId) {
        SysAiProviderKey key = keyMapper.selectById(keyId);
        if (key == null || !key.getProviderId().equals(providerId)) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "API Key 不存在");
        }
        return key;
    }

    private ProviderVO toVO(SysAiProvider provider) {
        ProviderVO vo = new ProviderVO();
        vo.setId(provider.getId());
        vo.setProviderCode(provider.getProviderCode());
        vo.setDisplayName(provider.getDisplayName());
        vo.setApiBaseUrl(provider.getApiBaseUrl());
        vo.setProtocolType(provider.getProtocolType());
        vo.setAuthType(provider.getAuthType());
        vo.setDefaultHeaders(AiJsonUtils.read(provider.getDefaultHeaders(),
                new TypeReference<Map<String, Object>>() {
                }));
        vo.setSortOrder(provider.getSortOrder());
        vo.setHealthCheckEnabled(provider.getHealthCheckEnabled());
        vo.setUserIdentityForward(AiJsonUtils.read(provider.getUserIdentityForward(),
                UserIdentityForwardForm.class));
        vo.setRemark(provider.getRemark());
        vo.setStatus(provider.getStatus());
        vo.setCreateTime(provider.getCreateTime());
        vo.setUpdateTime(provider.getUpdateTime());
        return vo;
    }

    private ProviderKeyVO toKeyVO(SysAiProviderKey key) {
        ProviderKeyVO vo = new ProviderKeyVO();
        vo.setId(key.getId());
        vo.setProviderId(key.getProviderId());
        vo.setName(key.getName());
        vo.setKeyPrefix(key.getKeyPrefix());
        vo.setStatus(key.getStatus());
        vo.setPriority(key.getPriority());
        vo.setWeight(key.getWeight());
        vo.setDailyQuota(key.getDailyQuota());
        vo.setRpmLimit(key.getRpmLimit());
        vo.setExpiresAt(key.getExpiresAt());
        vo.setLastUsedAt(key.getLastUsedAt());
        vo.setLastUsedBy(key.getLastUsedBy());
        vo.setCreateTime(key.getCreateTime());
        vo.setUpdateTime(key.getUpdateTime());
        return vo;
    }

    /** 失效启用供应商缓存：删 L2 同键 + 广播 cache:invalidation，同步清理 python 各实例 L1 */
    private void clearProviderCache() {
        redis.delete(PROVIDER_LIST_CACHE_KEY);
        try {
            Map<String, Object> message = new LinkedHashMap<>();
            message.put("type", "key");
            message.put("key", PROVIDER_LIST_CACHE_KEY);
            message.put("senderId", INSTANCE_ID);
            redis.convertAndSend(CACHE_INVALIDATION_CHANNEL, objectMapper.writeValueAsString(message));
        } catch (Exception e) {
            log.warn("启用供应商缓存失效广播失败: {}", e.getMessage());
        }
    }
}

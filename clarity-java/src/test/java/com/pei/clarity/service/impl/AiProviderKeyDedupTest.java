package com.pei.clarity.service.impl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.conditions.AbstractWrapper;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.baomidou.mybatisplus.core.toolkit.GlobalConfigUtils;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.clarity.common.exception.BusinessException;
import com.pei.clarity.common.result.ResultCode;
import com.pei.clarity.common.util.AiCredentialCipher;
import com.pei.clarity.mapper.SysAiModelMapper;
import com.pei.clarity.mapper.SysAiProviderKeyMapper;
import com.pei.clarity.mapper.SysAiProviderMapper;
import com.pei.clarity.model.entity.SysAiProvider;
import com.pei.clarity.model.entity.SysAiProviderKey;
import com.pei.clarity.model.form.ProviderKeyForm;
import com.pei.clarity.service.AiProviderHealthService;
import java.util.ArrayList;
import java.util.List;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.test.util.ReflectionTestUtils;

/**
 * API Key 查重按供应商维度消歧：唯一键为 (provider_id, key_hash)，同一把密钥（如本地三服务共用的
 * INTERNAL_SERVICE_KEY）登记到不同供应商必须放行，只有同一供应商内重复才拒绝。
 */
@DisplayName("AiProviderServiceImpl API Key 查重（provider 维度）")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiProviderKeyDedupTest {

    private static final String KEY_HASH = "HASH_OF_INTERNAL_SERVICE_KEY";

    @Mock
    private SysAiProviderKeyMapper keyMapper;

    @Mock
    private SysAiProviderMapper providerMapper;

    @Mock
    private SysAiModelMapper modelMapper;

    @Mock
    private AiProviderHealthService providerHealthService;

    @Mock
    private AiCredentialCipher cipher;

    @Mock
    private StringRedisTemplate redis;

    @Mock
    private ObjectMapper objectMapper;

    private AiProviderServiceImpl service;

    @BeforeAll
    static void initTableInfo() {
        MybatisConfiguration configuration = new MybatisConfiguration();
        GlobalConfigUtils.setGlobalConfig(configuration, GlobalConfigUtils.defaults());
        TableInfoHelper.initTableInfo(new MapperBuilderAssistant(configuration, ""), SysAiProviderKey.class);
    }

    @BeforeEach
    void setUp() {
        service = new AiProviderServiceImpl(keyMapper, modelMapper, providerHealthService, cipher, redis, objectMapper);
        ReflectionTestUtils.setField(service, "baseMapper", providerMapper);
        when(cipher.hashKey(any())).thenReturn(KEY_HASH);
        when(cipher.encrypt(any())).thenReturn("cipher");
        when(cipher.maskKey(any())).thenReturn("sk-abc...");
    }

    @Test
    @DisplayName("查重条件同时含 provider_id 与 key_hash（跨供应商不误拒的机制保证）")
    void dedupQueryScopedToProvider() {
        when(providerMapper.selectById(7L)).thenReturn(provider(7L));
        List<Object> wrappers = new ArrayList<>();
        when(keyMapper.selectCount(any())).thenAnswer(invocation -> {
            wrappers.add(invocation.getArgument(0));
            return 0L;
        });

        service.createKey(7L, keyForm("共用密钥"));

        AbstractWrapper<?, ?, ?> wrapper = (AbstractWrapper<?, ?, ?>) wrappers.get(0);
        assertThat(wrapper.getSqlSegment()).contains("provider_id").contains("key_hash");
        assertThat(wrapper.getParamNameValuePairs().values()).contains(7L, KEY_HASH);
    }

    @Test
    @DisplayName("同一供应商重复登记同一把密钥才拒绝，文案与 python 一致")
    void duplicateInSameProviderRejected() {
        when(providerMapper.selectById(7L)).thenReturn(provider(7L));
        when(keyMapper.selectCount(any())).thenReturn(1L);

        assertThatThrownBy(() -> service.createKey(7L, keyForm("重复密钥")))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("该供应商下该 API Key 已存在")
                .extracting(ex -> ((BusinessException) ex).getResultCode())
                .isEqualTo(ResultCode.DATA_EXISTS);
    }

    private static SysAiProvider provider(Long id) {
        SysAiProvider provider = new SysAiProvider();
        provider.setId(id);
        provider.setProviderCode("local-voice");
        provider.setDisplayName("本地语音");
        return provider;
    }

    private static ProviderKeyForm keyForm(String name) {
        ProviderKeyForm form = new ProviderKeyForm();
        form.setName(name);
        form.setKey("INTERNAL_SERVICE_KEY");
        return form;
    }
}

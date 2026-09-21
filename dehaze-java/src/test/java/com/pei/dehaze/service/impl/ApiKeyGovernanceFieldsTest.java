package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.baomidou.mybatisplus.core.toolkit.GlobalConfigUtils;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAiModelMapper;
import com.pei.dehaze.mapper.SysApiKeyMapper;
import com.pei.dehaze.model.dto.ApiKeyResult;
import com.pei.dehaze.model.entity.SysAiModel;
import com.pei.dehaze.model.entity.SysApiKey;
import com.pei.dehaze.model.form.ApiKeyForm;
import com.pei.dehaze.service.SysUserService;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * API Key 治理字段 CRUD 与白名单存在性校验（对齐 python {@code ApiKeyCreate}/{@code ApiKeyResult}/{@code _validate_whitelist}）。
 *
 * <p>治理字段 dailyQuota/monthlyQuota/rpmLimit/modelWhitelist 必须"落库 + 创建回显 + 列表回显"三处齐全；
 * 白名单内每个模型标识必须存在且启用（python 同口径，否则 A0400）——执行点不在 java
 * （兼容调用经 python 兼容层预检，java 转发即受治），此处只保证字段与校验不丢。
 */
@DisplayName("ApiKeyServiceImpl 治理字段 CRUD 与白名单校验")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class ApiKeyGovernanceFieldsTest {

    @Mock
    private SysUserService sysUserService;

    @Mock
    private SysApiKeyMapper sysApiKeyMapper;

    @Mock
    private SysAiModelMapper sysAiModelMapper;

    private ApiKeyServiceImpl service;

    /** 无 Spring 上下文时 LambdaQueryWrapper 的 eq(...) 需要先注册实体表信息 */
    @BeforeAll
    static void initTableInfo() {
        MybatisConfiguration configuration = new MybatisConfiguration();
        GlobalConfigUtils.setGlobalConfig(configuration, GlobalConfigUtils.defaults());
        MapperBuilderAssistant assistant = new MapperBuilderAssistant(configuration, "");
        TableInfoHelper.initTableInfo(assistant, SysApiKey.class);
        TableInfoHelper.initTableInfo(assistant, SysAiModel.class);
    }

    @BeforeEach
    void setUp() {
        service = new ApiKeyServiceImpl(sysUserService, sysApiKeyMapper, sysAiModelMapper);
        ReflectionTestUtils.setField(service, "baseMapper", sysApiKeyMapper);
    }

    @Test
    @DisplayName("创建：模型白名单落库并原样回显")
    void createEchoesModelWhitelist() {
        when(sysAiModelMapper.selectCount(any())).thenReturn(1L);
        ApiKeyForm form = new ApiKeyForm();
        form.setName("白名单密钥");
        form.setDailyQuota(1000L);
        form.setMonthlyQuota(20000L);
        form.setRpmLimit(60L);
        form.setModelWhitelist(List.of("qwen3-0.6b", "gpt-4o"));

        ApiKeyResult result = service.createApiKey(form);

        assertThat(result.getModelWhitelist()).containsExactly("qwen3-0.6b", "gpt-4o");
        assertThat(result.getDailyQuota()).isEqualTo(1000L);
        assertThat(result.getMonthlyQuota()).isEqualTo(20000L);
        assertThat(result.getRpmLimit()).isEqualTo(60L);
        verify(sysApiKeyMapper).insert(any());
    }

    @Test
    @DisplayName("创建：空数组与不传同义（落 NULL 而非 JSON 空数组）")
    void createWithEmptyWhitelistFallsBackToNull() {
        ApiKeyForm form = new ApiKeyForm();
        form.setName("空数组密钥");
        form.setModelWhitelist(List.of());

        ApiKeyResult result = service.createApiKey(form);

        assertThat(result.getModelWhitelist()).isNull();
    }

    @ParameterizedTest(name = "白名单含不可用模型 [{0}] → A0400 且不落库")
    @ValueSource(strings = {"no_such_model", "disabled_model", "deleted_model", "wl_ok "})
    @DisplayName("创建：白名单内模型不存在/未启用一律 A0400")
    void createRejectsUnavailableWhitelistModel(String modelId) {
        when(sysAiModelMapper.selectCount(any())).thenReturn(0L);
        ApiKeyForm form = new ApiKeyForm();
        form.setName("非法白名单密钥");
        form.setModelWhitelist(List.of(modelId));

        BusinessException ex = assertThrows(BusinessException.class, () -> service.createApiKey(form));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.PARAM_ERROR);
        assertThat(ex.getMessage()).contains(modelId + " 不存在或未启用");
        verify(sysApiKeyMapper, never()).insert(any());
    }

    @Test
    @DisplayName("存在性查询必须带 status=1（已禁用模型不可入白名单）")
    void whitelistQueryFiltersEnabledOnly() {
        when(sysAiModelMapper.selectCount(any())).thenReturn(1L);
        ApiKeyForm form = new ApiKeyForm();
        form.setName("查询口径密钥");
        form.setModelWhitelist(List.of("qwen3-0.6b"));

        service.createApiKey(form);

        ArgumentCaptor<LambdaQueryWrapper<SysAiModel>> captor = ArgumentCaptor.forClass(LambdaQueryWrapper.class);
        verify(sysAiModelMapper).selectCount(captor.capture());
        assertThat(captor.getValue().getSqlSegment()).contains("model_id").contains("status");
        assertThat(captor.getValue().getParamNameValuePairs().values()).contains("qwen3-0.6b", 1);
    }

    @Test
    @DisplayName("混合白名单：合法 + 非法整体拒绝（逐个校验，不因首个合法短路放行）")
    void createRejectsMixedWhitelist() {
        when(sysAiModelMapper.selectCount(any())).thenReturn(1L, 0L);
        ApiKeyForm form = new ApiKeyForm();
        form.setName("混合白名单密钥");
        form.setModelWhitelist(List.of("wl_ok", "wl_missing"));

        BusinessException ex = assertThrows(BusinessException.class, () -> service.createApiKey(form));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.PARAM_ERROR);
        assertThat(ex.getMessage()).contains("wl_missing 不存在或未启用");
        verify(sysApiKeyMapper, never()).insert(any());
    }

    @Test
    @DisplayName("列表：四个治理字段随列表回显（python ApiKeyResult 同形态）")
    void listEchoesGovernanceFields() {
        SysApiKey key = new SysApiKey();
        key.setId(1L);
        key.setName("白名单密钥");
        key.setKeyPrefix("dhak_abc");
        key.setStatus(1);
        key.setDailyQuota(1000L);
        key.setMonthlyQuota(20000L);
        key.setRpmLimit(60L);
        key.setModelWhitelist(List.of("qwen3-0.6b"));
        when(sysApiKeyMapper.selectList(any())).thenReturn(List.of(key));

        List<ApiKeyResult> list = service.listApiKeys();

        assertThat(list).hasSize(1);
        assertThat(list.get(0).getDailyQuota()).isEqualTo(1000L);
        assertThat(list.get(0).getMonthlyQuota()).isEqualTo(20000L);
        assertThat(list.get(0).getRpmLimit()).isEqualTo(60L);
        assertThat(list.get(0).getModelWhitelist()).containsExactly("qwen3-0.6b");
    }
}

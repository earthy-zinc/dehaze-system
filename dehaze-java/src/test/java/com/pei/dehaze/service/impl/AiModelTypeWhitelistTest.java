package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.baomidou.mybatisplus.core.toolkit.GlobalConfigUtils;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAiModelMapper;
import com.pei.dehaze.mapper.SysAiModelPriceDetailMapper;
import com.pei.dehaze.mapper.SysAiModelPriceMapper;
import com.pei.dehaze.mapper.SysMemberMapper;
import com.pei.dehaze.model.entity.SysAiModel;
import com.pei.dehaze.model.form.AiModelForm;
import com.pei.dehaze.model.form.AiModelUpdateForm;
import com.pei.dehaze.model.query.AiModelPageQuery;
import com.pei.dehaze.service.AiProviderHealthService;
import com.pei.dehaze.service.MessageService;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.test.util.ReflectionTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * AI 模型 {@code model_type} 白名单（service 层单一信息源，对齐 python {@code Literal["chat","embedding","rerank"]}）。
 *
 * <p>python 由 pydantic 在请求校验阶段拒绝非法字面量（A0400），覆盖创建/更新/列表筛选三处；
 * java 若不在 service 层拦，拼错的类型会直接落库，或把非法筛选值静默当成"无匹配数据"。
 */
@DisplayName("AiModelServiceImpl modelType 白名单")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiModelTypeWhitelistTest {

    @Mock
    private SysAiModelMapper modelMapper;

    @Mock
    private SysAiModelPriceMapper priceMapper;

    @Mock
    private SysAiModelPriceDetailMapper priceDetailMapper;

    @Mock
    private SysMemberMapper memberMapper;

    @Mock
    private AiProviderHealthService providerHealthService;

    @Mock
    private MessageService messageService;

    @Mock
    private StringRedisTemplate redis;

    @Mock
    private ObjectMapper objectMapper;

    private AiModelServiceImpl service;

    /** 无 Spring 上下文时 LambdaQueryWrapper 的 eq(...) 需要先注册实体表信息 */
    @BeforeAll
    static void initTableInfo() {
        MybatisConfiguration configuration = new MybatisConfiguration();
        GlobalConfigUtils.setGlobalConfig(configuration, GlobalConfigUtils.defaults());
        TableInfoHelper.initTableInfo(new MapperBuilderAssistant(configuration, ""), SysAiModel.class);
    }

    @BeforeEach
    void setUp() {
        service = new AiModelServiceImpl(priceMapper, priceDetailMapper, memberMapper,
                providerHealthService, messageService, redis, objectMapper);
        ReflectionTestUtils.setField(service, "baseMapper", modelMapper);
    }

    @ParameterizedTest(name = "modelType=[{0}] → A0400 且不触达 mapper")
    @NullSource
    @ValueSource(strings = {"CHAT", "llm", "rerank/embedding", "chat ", "", "chat;drop", "聊天"})
    @DisplayName("createModel：白名单外类型一律 A0400，不得落库")
    void createModelRejectsTypeOutsideWhitelist(String modelType) {
        AiModelForm form = new AiModelForm();
        form.setProviderId(1L);
        form.setModelId("whitelist_probe");
        form.setDisplayName("白名单探针");
        form.setModelType(modelType);

        BusinessException ex = assertThrows(BusinessException.class, () -> service.createModel(form));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.PARAM_ERROR);
        verifyNoInteractions(modelMapper, priceMapper, priceDetailMapper, memberMapper,
                providerHealthService, messageService, redis, objectMapper);
    }

    @Test
    @DisplayName("updateModel：白名单外类型 A0400（先于'创建后不可修改'的业务拒绝）")
    void updateModelRejectsTypeOutsideWhitelist() {
        AiModelUpdateForm form = new AiModelUpdateForm();
        form.setModelType("bogus");

        BusinessException ex = assertThrows(BusinessException.class, () -> service.updateModel("m1", form));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.PARAM_ERROR);
        verifyNoInteractions(modelMapper);
    }

    @Test
    @DisplayName("白名单内取值不误伤：embedding 通过白名单后落到缺 dimension 的单字段校验")
    void createModelAcceptsWhitelistedType() {
        AiModelForm form = new AiModelForm();
        form.setProviderId(1L);
        form.setModelId("whitelist_probe_legal");
        form.setDisplayName("合法类型");
        form.setModelType("embedding");

        BusinessException ex = assertThrows(BusinessException.class, () -> service.createModel(form));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.PARAM_ERROR);
        assertThat(ex.getMessage()).contains("dimension");
        verifyNoInteractions(modelMapper);
    }

    @ParameterizedTest(name = "modelType=[{0}] 筛选 → A0400 且不触达 mapper")
    @ValueSource(strings = {"CHAT", "llm", "chat ", "rerank/embedding", "聊天", ""})
    @DisplayName("listModels：非法筛选值 A0400（而非静默返回空列表）")
    void listModelsRejectsTypeOutsideWhitelist(String modelType) {
        AiModelPageQuery query = new AiModelPageQuery();
        query.setModelType(modelType);

        BusinessException ex = assertThrows(BusinessException.class, () -> service.listModels(query));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.PARAM_ERROR);
        verifyNoInteractions(modelMapper);
    }

    @Test
    @DisplayName("listModels：不传与三个合法值均放行（白名单不误伤）")
    void listModelsAcceptsWhitelistedType() {
        // 回填传入的 page 实例（空记录），避免泛型桩返回类型不匹配
        when(modelMapper.selectPage(any(), any()))
                .thenAnswer(invocation -> invocation.getArgument(0));

        for (String modelType : new String[]{null, "chat", "embedding", "rerank"}) {
            AiModelPageQuery query = new AiModelPageQuery();
            query.setModelType(modelType);
            query.setPageNum(1);
            query.setPageSize(10);

            assertThat(service.listModels(query)).isNotNull();
        }
    }
}

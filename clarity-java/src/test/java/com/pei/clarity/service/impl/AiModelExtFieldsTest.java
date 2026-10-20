package com.pei.clarity.service.impl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.conditions.AbstractWrapper;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.baomidou.mybatisplus.core.toolkit.GlobalConfigUtils;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.clarity.mapper.SysAiModelMapper;
import com.pei.clarity.mapper.SysAiModelPriceDetailMapper;
import com.pei.clarity.mapper.SysAiModelPriceMapper;
import com.pei.clarity.model.entity.SysAiModel;
import com.pei.clarity.model.entity.SysAiModelPrice;
import com.pei.clarity.model.entity.SysAiModelPriceDetail;
import com.pei.clarity.model.form.AiModelForm;
import com.pei.clarity.model.form.AiModelUpdateForm;
import com.pei.clarity.model.form.ModelPriceForm;
import com.pei.clarity.model.query.AiModelPageQuery;
import com.pei.clarity.model.vo.AiModelVO;
import com.pei.clarity.model.vo.ModelPriceVO;
import com.pei.clarity.service.AiDefaultModelResolver;
import com.pei.clarity.service.AiProviderHealthService;
import com.pei.clarity.service.AiUserLevelResolver;
import com.pei.clarity.service.FavoriteService;
import com.pei.clarity.service.MessageService;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.test.util.ReflectionTestUtils;

/** 模型 task_tags 与价格 billing_unit/usage_kind 的落库与回显（全模态扩展新增字段）。 */
@DisplayName("AiModelServiceImpl 全模态扩展字段")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiModelExtFieldsTest {

    @Mock
    private SysAiModelMapper modelMapper;

    @Mock
    private SysAiModelPriceMapper priceMapper;

    @Mock
    private SysAiModelPriceDetailMapper priceDetailMapper;

    @Mock
    private AiUserLevelResolver userLevelResolver;

    @Mock
    private AiProviderHealthService providerHealthService;

    @Mock
    private MessageService messageService;

    @Mock
    private FavoriteService favoriteService;

    @Mock
    private AiDefaultModelResolver aiDefaultModelResolver;

    @Mock
    private StringRedisTemplate redis;

    private final ObjectMapper objectMapper = new ObjectMapper();

    private AiModelServiceImpl service;

    @BeforeAll
    static void initTableInfo() {
        MybatisConfiguration configuration = new MybatisConfiguration();
        GlobalConfigUtils.setGlobalConfig(configuration, GlobalConfigUtils.defaults());
        TableInfoHelper.initTableInfo(new MapperBuilderAssistant(configuration, ""), SysAiModel.class);
    }

    @BeforeEach
    void setUp() {
        service = new AiModelServiceImpl(
                priceMapper,
                priceDetailMapper,
                userLevelResolver,
                providerHealthService,
                messageService,
                favoriteService,
                redis,
                objectMapper,
                aiDefaultModelResolver);
        ReflectionTestUtils.setField(service, "baseMapper", modelMapper);
    }

    @Test
    @DisplayName("createModel：taskTags 序列化为 JSON 数组落库，并按 string[] 回显")
    void createModelPersistsTaskTags() {
        AiModelForm form = new AiModelForm();
        form.setProviderId(1L);
        form.setModelId("aecrnet");
        form.setDisplayName("AECRNet 去雾");
        form.setModelType("image_edit");
        form.setTaskTags(List.of("dehaze"));

        AiModelVO vo = service.createModel(form);

        ArgumentCaptor<SysAiModel> captor = ArgumentCaptor.forClass(SysAiModel.class);
        org.mockito.Mockito.verify(modelMapper).insert(captor.capture());
        assertThat(captor.getValue().getTaskTags()).isEqualTo("[\"dehaze\"]");
        assertThat(vo.getTaskTags()).containsExactly("dehaze");
    }

    @Test
    @DisplayName("createModel：未传 taskTags 落 NULL")
    void createModelWithoutTaskTagsPersistsNull() {
        AiModelForm form = new AiModelForm();
        form.setProviderId(1L);
        form.setModelId("plain");
        form.setDisplayName("无标签模型");
        form.setModelType("chat");

        service.createModel(form);

        ArgumentCaptor<SysAiModel> captor = ArgumentCaptor.forClass(SysAiModel.class);
        org.mockito.Mockito.verify(modelMapper).insert(captor.capture());
        assertThat(captor.getValue().getTaskTags()).isNull();
    }

    @Test
    @DisplayName("updateModel：taskTags 全量替换（空数组清空）")
    void updateModelReplacesTaskTags() {
        SysAiModel existing = new SysAiModel();
        existing.setId(7L);
        existing.setModelId("sensevoice");
        existing.setModelType("asr");
        existing.setStatus(1);
        existing.setTaskTags("[\"stream\"]");
        when(modelMapper.selectList(any())).thenReturn(List.of(existing));

        AiModelUpdateForm form = new AiModelUpdateForm();
        form.setTaskTags(List.of("offline"));

        service.updateModel("sensevoice", form);

        ArgumentCaptor<SysAiModel> captor = ArgumentCaptor.forClass(SysAiModel.class);
        org.mockito.Mockito.verify(modelMapper).updateById(captor.capture());
        assertThat(captor.getValue().getTaskTags()).isEqualTo("[\"offline\"]");
    }

    @Test
    @DisplayName("createPrice：billing_unit 与 detail.usage_kind 落库并回显")
    void createPricePersistsBillingUnitAndUsageKind() {
        ModelPriceForm form = new ModelPriceForm();
        form.setModelId("sensevoice");
        form.setProviderId(2L);
        form.setBillingUnit("second");
        ModelPriceForm.Detail detail = new ModelPriceForm.Detail();
        detail.setTokenType("unit");
        detail.setUsageKind("unit");
        detail.setTimeSlot("peak");
        detail.setUnitPrice(new BigDecimal("0.5000"));
        form.setDetails(List.of(detail));

        ModelPriceVO vo = service.createPrice(form);

        ArgumentCaptor<SysAiModelPrice> priceCaptor = ArgumentCaptor.forClass(SysAiModelPrice.class);
        org.mockito.Mockito.verify(priceMapper).insert(priceCaptor.capture());
        assertThat(priceCaptor.getValue().getBillingUnit()).isEqualTo("second");

        ArgumentCaptor<SysAiModelPriceDetail> detailCaptor = ArgumentCaptor.forClass(SysAiModelPriceDetail.class);
        org.mockito.Mockito.verify(priceDetailMapper).insert(detailCaptor.capture());
        assertThat(detailCaptor.getValue().getUsageKind()).isEqualTo("unit");

        assertThat(vo.getBillingUnit()).isEqualTo("second");
        assertThat(vo.getDetails()).hasSize(1);
        assertThat(vo.getDetails().get(0).getUsageKind()).isEqualTo("unit");
    }

    @Test
    @DisplayName("listModels：taskTags 逗号分隔落为 JSON_OVERLAPS 数组参数（去空白、OR 语义）")
    void listModelsAppliesTaskTagsFilter() {
        List<Object> wrappers = new ArrayList<>();
        when(modelMapper.selectPage(any(), any())).thenAnswer(invocation -> {
            wrappers.add(invocation.getArgument(1));
            return invocation.getArgument(0);
        });

        AiModelPageQuery query = new AiModelPageQuery();
        query.setTaskTags(" clarity , stream , ");
        query.setPageNum(1);
        query.setPageSize(10);

        service.listModels(query);

        AbstractWrapper<?, ?, ?> wrapper = (AbstractWrapper<?, ?, ?>) wrappers.get(0);
        // apply 的 SQL 与参数在 getSqlSegment() 惰性求值时才会写入 paramNameValuePairs，故先取 segment
        assertThat(wrapper.getSqlSegment()).contains("JSON_OVERLAPS(task_tags");
        assertThat(wrapper.getParamNameValuePairs().values()).contains("[\"dehaze\",\"stream\"]");
    }

    @Test
    @DisplayName("listModels：taskTags 为空/空白时不追加筛选条件")
    void listModelsSkipsBlankTaskTags() {
        List<Object> wrappers = new ArrayList<>();
        when(modelMapper.selectPage(any(), any())).thenAnswer(invocation -> {
            wrappers.add(invocation.getArgument(1));
            return invocation.getArgument(0);
        });

        AiModelPageQuery query = new AiModelPageQuery();
        query.setTaskTags(" , ");
        query.setPageNum(1);
        query.setPageSize(10);

        service.listModels(query);

        AbstractWrapper<?, ?, ?> wrapper = (AbstractWrapper<?, ?, ?>) wrappers.get(0);
        assertThat(wrapper.getSqlSegment()).doesNotContain("JSON_OVERLAPS");
    }

    @ParameterizedTest(name = "modelType=[{0}] → billingUnit=[{1}] usageKind=[{2}]")
    @CsvSource({
        "asr, second, unit",
        "tts, ten_thousand_chars, unit",
        "image_edit, image, unit",
        "image_gen, image, unit",
        "video_gen, frame, unit",
        "chat, million_tokens, token",
        "embedding, million_tokens, token",
        "music_gen, million_tokens, token",
        "3d_gen, million_tokens, token"
    })
    @DisplayName("createPrice：未传 billingUnit/usageKind 时按模型类型派生（不无条件写死 million_tokens）")
    void createPriceDerivesBillingUnitFromModelType(String modelType, String expectUnit, String expectUsageKind) {
        SysAiModel owner = new SysAiModel();
        owner.setId(5L);
        owner.setModelId("derive_probe");
        owner.setProviderId(2L);
        owner.setModelType(modelType);
        when(modelMapper.selectList(any())).thenReturn(List.of(owner));
        when(modelMapper.selectOne(any())).thenReturn(owner);

        ModelPriceForm form = new ModelPriceForm();
        form.setModelId("derive_probe");
        form.setProviderId(2L);
        ModelPriceForm.Detail detail = new ModelPriceForm.Detail();
        detail.setTokenType("input");
        detail.setTimeSlot("peak");
        detail.setUnitPrice(new BigDecimal("0.5000"));
        form.setDetails(List.of(detail));

        ModelPriceVO vo = service.createPrice(form);

        ArgumentCaptor<SysAiModelPrice> priceCaptor = ArgumentCaptor.forClass(SysAiModelPrice.class);
        org.mockito.Mockito.verify(priceMapper).insert(priceCaptor.capture());
        assertThat(priceCaptor.getValue().getBillingUnit()).isEqualTo(expectUnit);

        ArgumentCaptor<SysAiModelPriceDetail> detailCaptor = ArgumentCaptor.forClass(SysAiModelPriceDetail.class);
        org.mockito.Mockito.verify(priceDetailMapper).insert(detailCaptor.capture());
        assertThat(detailCaptor.getValue().getUsageKind()).isEqualTo(expectUsageKind);

        assertThat(vo.getBillingUnit()).isEqualTo(expectUnit);
        assertThat(vo.getDetails().get(0).getUsageKind()).isEqualTo(expectUsageKind);
    }

    @Test
    @DisplayName("createPrice：显式传入的 billingUnit/usageKind 优先于派生值")
    void explicitBillingUnitWins() {
        SysAiModel owner = new SysAiModel();
        owner.setId(6L);
        owner.setModelId("explicit_probe");
        owner.setProviderId(2L);
        owner.setModelType("asr");
        when(modelMapper.selectList(any())).thenReturn(List.of(owner));
        when(modelMapper.selectOne(any())).thenReturn(owner);

        ModelPriceForm form = new ModelPriceForm();
        form.setModelId("explicit_probe");
        form.setProviderId(2L);
        form.setBillingUnit("million_tokens");
        ModelPriceForm.Detail detail = new ModelPriceForm.Detail();
        detail.setTokenType("input");
        detail.setUsageKind("token");
        detail.setTimeSlot("peak");
        detail.setUnitPrice(new BigDecimal("1.0000"));
        form.setDetails(List.of(detail));

        ModelPriceVO vo = service.createPrice(form);

        assertThat(vo.getBillingUnit()).isEqualTo("million_tokens");
        assertThat(vo.getDetails().get(0).getUsageKind()).isEqualTo("token");
    }
}

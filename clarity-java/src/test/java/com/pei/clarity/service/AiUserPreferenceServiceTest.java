package com.pei.clarity.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.pei.clarity.common.exception.BusinessException;
import com.pei.clarity.common.result.ResultCode;
import com.pei.clarity.mapper.SysAiAgentMapper;
import com.pei.clarity.mapper.SysAiAgentVersionMapper;
import com.pei.clarity.mapper.SysAiModelMapper;
import com.pei.clarity.mapper.SysAiUserPreferenceMapper;
import com.pei.clarity.mapper.SysDictMapper;
import com.pei.clarity.model.entity.SysAiModel;
import com.pei.clarity.model.entity.SysAiUserPreference;
import com.pei.clarity.model.entity.SysDict;
import com.pei.clarity.model.form.AiUserPreferenceForm;
import com.pei.clarity.model.vo.AiUserPreferenceVO;
import com.pei.clarity.service.impl.AiUserPreferenceServiceImpl;
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

/**
 * 用户偏好默认模型读写：GET 不建行、PUT 校验后 upsert、null 清除。
 *
 * <p>装配真实 {@link AiDefaultModelResolver}（mapper 为 mock），以覆盖"校验不通过 → A0601 不落库"的接线。
 */
@DisplayName("AiUserPreferenceService 用户偏好默认模型")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiUserPreferenceServiceTest {

    private static final Long USER_ID = 11L;

    @Mock
    private SysAiUserPreferenceMapper preferenceMapper;

    @Mock
    private SysAiModelMapper modelMapper;

    @Mock
    private SysDictMapper dictMapper;

    @Mock
    private SysAiAgentMapper agentMapper;

    @Mock
    private SysAiAgentVersionMapper agentVersionMapper;

    @Mock
    private AiUserLevelResolver userLevelResolver;

    private AiUserPreferenceServiceImpl service;

    @BeforeAll
    static void initMybatisPlusLambdaCache() {
        initTableInfo(SysAiUserPreference.class, "com.pei.clarity.mapper.SysAiUserPreferenceMapper");
        initTableInfo(SysAiModel.class, "com.pei.clarity.mapper.SysAiModelMapper");
        initTableInfo(SysDict.class, "com.pei.clarity.mapper.SysDictMapper");
    }

    private static void initTableInfo(Class<?> entity, String namespace) {
        MybatisConfiguration configuration = new MybatisConfiguration();
        MapperBuilderAssistant assistant = new MapperBuilderAssistant(configuration, "user-preference-test");
        assistant.setCurrentNamespace(namespace);
        TableInfoHelper.initTableInfo(assistant, entity);
    }

    @BeforeEach
    void setUp() {
        AiDefaultModelResolver resolver = new AiDefaultModelResolver(
                preferenceMapper, modelMapper, dictMapper, agentMapper, agentVersionMapper, userLevelResolver);
        service = new AiUserPreferenceServiceImpl(preferenceMapper, resolver);
    }

    private void stubPreference(String defaultModel) {
        SysAiUserPreference preference = new SysAiUserPreference();
        preference.setUserId(USER_ID);
        preference.setDefaultModel(defaultModel);
        when(preferenceMapper.selectOne(any(Wrapper.class))).thenReturn(preference);
    }

    private SysAiModel chatModel(String modelId, Integer vipLevel) {
        SysAiModel model = new SysAiModel();
        model.setModelId(modelId);
        model.setModelType("chat");
        model.setStatus(1);
        model.setVipLevel(vipLevel);
        return model;
    }

    @Test
    @DisplayName("GET：无记录返回 defaultModel=null，且不自动建行")
    void getWithoutRecordReturnsNullAndCreatesNothing() {
        when(preferenceMapper.selectOne(any(Wrapper.class))).thenReturn(null);

        AiUserPreferenceVO vo = service.get(USER_ID);

        assertThat(vo.getDefaultModel()).isNull();
        verify(preferenceMapper, never()).upsertDefaultModel(any(), any());
        verify(preferenceMapper, never()).insert(any(SysAiUserPreference.class));
    }

    @Test
    @DisplayName("GET：有记录但值为空返回 null")
    void getWithBlankValueReturnsNull() {
        stubPreference(null);

        assertThat(service.get(USER_ID).getDefaultModel()).isNull();
    }

    @Test
    @DisplayName("GET：返回已存偏好模型")
    void getReturnsStoredModel() {
        stubPreference("qwen3-0.6b");

        assertThat(service.get(USER_ID).getDefaultModel()).isEqualTo("qwen3-0.6b");
    }

    @Test
    @DisplayName("PUT：设置可用模型后可读回")
    void updateStoresSelectableModel() {
        when(modelMapper.selectOne(any(Wrapper.class))).thenReturn(chatModel("qwen3-0.6b", 0));
        when(userLevelResolver.levelOf(USER_ID)).thenReturn(0);
        AiUserPreferenceForm form = new AiUserPreferenceForm();
        form.setDefaultModel("qwen3-0.6b");

        service.update(USER_ID, form);

        verify(preferenceMapper).upsertDefaultModel(USER_ID, "qwen3-0.6b");
        stubPreference("qwen3-0.6b");
        assertThat(service.get(USER_ID).getDefaultModel()).isEqualTo("qwen3-0.6b");
    }

    @Test
    @DisplayName("PUT：defaultModel=null 清除偏好，不做模型校验")
    void updateWithNullClearsPreference() {
        AiUserPreferenceForm form = new AiUserPreferenceForm();

        service.update(USER_ID, form);

        verify(preferenceMapper).upsertDefaultModel(eq(USER_ID), isNull());
        verify(modelMapper, never()).selectOne(any(Wrapper.class));
    }

    @Test
    @DisplayName("PUT：模型不存在/已停用/非 chat 报 A0601 且不落库")
    void updateWithUnavailableModelRejected() {
        when(modelMapper.selectOne(any(Wrapper.class))).thenReturn(null);
        AiUserPreferenceForm form = new AiUserPreferenceForm();
        form.setDefaultModel("retired-model");

        BusinessException ex = assertThrows(BusinessException.class, () -> service.update(USER_ID, form));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.AI_MODEL_NOT_AVAILABLE);
        assertThat(ex.getMessage()).isEqualTo("模型不存在、未启用或当前等级不可用");
        verify(preferenceMapper, never()).upsertDefaultModel(any(), any());
    }

    @Test
    @DisplayName("PUT：模型 vip_level 超出用户等级报 A0601 且不落库")
    void updateWithModelAboveUserLevelRejected() {
        when(modelMapper.selectOne(any(Wrapper.class))).thenReturn(chatModel("gpt-4o", 3));
        when(userLevelResolver.levelOf(USER_ID)).thenReturn(1);
        AiUserPreferenceForm form = new AiUserPreferenceForm();
        form.setDefaultModel("gpt-4o");

        BusinessException ex = assertThrows(BusinessException.class, () -> service.update(USER_ID, form));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.AI_MODEL_NOT_AVAILABLE);
        assertThat(ex.getMessage()).isEqualTo("模型不存在、未启用或当前等级不可用");
        verify(preferenceMapper, never()).upsertDefaultModel(any(), any());
    }
}

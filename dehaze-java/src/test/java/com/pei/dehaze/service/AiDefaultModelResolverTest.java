package com.pei.dehaze.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.conditions.AbstractWrapper;
import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAiAgentMapper;
import com.pei.dehaze.mapper.SysAiAgentVersionMapper;
import com.pei.dehaze.mapper.SysAiModelMapper;
import com.pei.dehaze.mapper.SysAiUserPreferenceMapper;
import com.pei.dehaze.mapper.SysDictMapper;
import com.pei.dehaze.model.entity.SysAiAgent;
import com.pei.dehaze.model.entity.SysAiAgentVersion;
import com.pei.dehaze.model.entity.SysAiModel;
import com.pei.dehaze.model.entity.SysAiUserPreference;
import com.pei.dehaze.model.entity.SysDict;
import java.util.ArrayList;
import java.util.Map;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

/**
 * 默认对话模型解析链：请求指定 ＞ 用户偏好 ＞ 平台默认字典 ＞ A0500（绝不回落代码常量）。
 *
 * <p>偏好模型的可用性（存在 + status=1 + model_type='chat'）由查询条件自身表达，
 * 纯单测无 SQL 执行，故以 wrapper 片段与参数断言其口径。
 */
@DisplayName("AiDefaultModelResolver 默认对话模型解析链")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiDefaultModelResolverTest {

    private static final Long USER_ID = 7L;

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

    private AiDefaultModelResolver resolver;

    @BeforeAll
    static void initMybatisPlusLambdaCache() {
        initTableInfo(SysAiUserPreference.class, "com.pei.dehaze.mapper.SysAiUserPreferenceMapper");
        initTableInfo(SysAiModel.class, "com.pei.dehaze.mapper.SysAiModelMapper");
        initTableInfo(SysDict.class, "com.pei.dehaze.mapper.SysDictMapper");
        initTableInfo(SysAiAgent.class, "com.pei.dehaze.mapper.SysAiAgentMapper");
        initTableInfo(SysAiAgentVersion.class, "com.pei.dehaze.mapper.SysAiAgentVersionMapper");
    }

    private static void initTableInfo(Class<?> entity, String namespace) {
        MybatisConfiguration configuration = new MybatisConfiguration();
        MapperBuilderAssistant assistant = new MapperBuilderAssistant(configuration, "default-model-test");
        assistant.setCurrentNamespace(namespace);
        TableInfoHelper.initTableInfo(assistant, entity);
    }

    @BeforeEach
    void setUp() {
        resolver = new AiDefaultModelResolver(
                preferenceMapper, modelMapper, dictMapper, agentMapper, agentVersionMapper, userLevelResolver);
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

    private SysDict platformDefault(String value) {
        SysDict dict = new SysDict();
        dict.setTypeCode("ai_model");
        dict.setName("default_model");
        dict.setStatus(1);
        dict.setValue(value);
        return dict;
    }

    @Test
    @DisplayName("用户偏好模型可用时取偏好，不读平台默认")
    void preferenceModelWins() {
        stubPreference("qwen3-0.6b");
        when(modelMapper.selectOne(any(Wrapper.class))).thenReturn(chatModel("qwen3-0.6b", 0));

        assertThat(resolver.resolveForUser(USER_ID, null)).isEqualTo("qwen3-0.6b");
        verify(dictMapper, never()).selectOne(any(Wrapper.class));
    }

    @Test
    @DisplayName("偏好模型已停用/非 chat（查询无命中）时静默跳过，回落平台默认")
    void unavailablePreferenceFallsBackToPlatformDefault() {
        stubPreference("retired-model");
        when(modelMapper.selectOne(any(Wrapper.class))).thenReturn(null);
        when(dictMapper.selectOne(any(Wrapper.class))).thenReturn(platformDefault("qwen3-0.6b"));

        assertThat(resolver.resolveForUser(USER_ID, null)).isEqualTo("qwen3-0.6b");
    }

    @Test
    @DisplayName("偏好为空值（等价未设置）时取平台默认字典值")
    void blankPreferenceFallsBackToPlatformDefault() {
        stubPreference(null);
        when(dictMapper.selectOne(any(Wrapper.class))).thenReturn(platformDefault("qwen3-0.6b"));

        assertThat(resolver.resolveForUser(USER_ID, null)).isEqualTo("qwen3-0.6b");
        verify(modelMapper, never()).selectOne(any(Wrapper.class));
    }

    @Test
    @DisplayName("无偏好记录时取平台默认字典值")
    void absentPreferenceUsesPlatformDefault() {
        when(preferenceMapper.selectOne(any(Wrapper.class))).thenReturn(null);
        when(dictMapper.selectOne(any(Wrapper.class))).thenReturn(platformDefault("qwen3-0.6b"));

        assertThat(resolver.resolveForUser(USER_ID, null)).isEqualTo("qwen3-0.6b");
    }

    @Test
    @DisplayName("平台默认未配置报 A0500，不回落代码常量")
    void platformDefaultMissingThrows() {
        when(preferenceMapper.selectOne(any(Wrapper.class))).thenReturn(null);
        when(dictMapper.selectOne(any(Wrapper.class))).thenReturn(null);

        BusinessException ex = assertThrows(BusinessException.class, () -> resolver.resolveForUser(USER_ID, null));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.BUSINESS_ERROR);
        assertThat(ex.getMessage()).isEqualTo("未配置平台默认模型（sys_dict: ai_model.default_model），请先在 AI 模型管理/字典管理中配置");
    }

    @Test
    @DisplayName("平台默认字典值为空白视同未配置，同样报 A0500")
    void blankPlatformDefaultThrows() {
        when(preferenceMapper.selectOne(any(Wrapper.class))).thenReturn(null);
        when(dictMapper.selectOne(any(Wrapper.class))).thenReturn(platformDefault("  "));

        BusinessException ex = assertThrows(BusinessException.class, () -> resolver.resolveForUser(USER_ID, null));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.BUSINESS_ERROR);
    }

    @Test
    @DisplayName("偏好模型可用性口径：查询条件强制 model_id + status=1 + model_type=chat")
    void preferenceLookupEnforcesEnabledChatModel() {
        stubPreference("qwen3-0.6b");
        when(modelMapper.selectOne(any(Wrapper.class))).thenReturn(chatModel("qwen3-0.6b", 0));

        resolver.resolveForUser(USER_ID, null);

        ArgumentCaptor<Wrapper<SysAiModel>> captor = ArgumentCaptor.forClass(Wrapper.class);
        verify(modelMapper).selectOne(captor.capture());
        AbstractWrapper<?, ?, ?> wrapper = (AbstractWrapper<?, ?, ?>) captor.getValue();
        assertThat(wrapper.getSqlSegment())
                .contains("model_id")
                .contains("status")
                .contains("model_type");
        assertThat(wrapper.getParamNameValuePairs().values()).contains("qwen3-0.6b", 1, "chat");
    }

    private SysAiAgent agent(String code, String mainModel) {
        SysAiAgent a = new SysAiAgent();
        a.setId(9L);
        a.setAgentCode(code);
        a.setModelId(mainModel);
        return a;
    }

    private SysAiAgentVersion publishedVersion(String snapshotModel) {
        SysAiAgentVersion v = new SysAiAgentVersion();
        v.setVersionNo(1);
        v.setStatus(2);
        v.setSnapshot(Map.of("model_id", snapshotModel));
        return v;
    }

    @Test
    @DisplayName("Agent 级：无偏好时取 Agent 已发布版本快照模型（不读平台默认）")
    void agentPublishedSnapshotModelWins() {
        when(preferenceMapper.selectOne(any(Wrapper.class))).thenReturn(null);
        when(agentMapper.selectOne(any(Wrapper.class))).thenReturn(agent("t-agent", "agent-main"));
        when(agentVersionMapper.selectOne(any(Wrapper.class))).thenReturn(publishedVersion("agent-published"));

        assertThat(resolver.resolveForUser(USER_ID, "t-agent")).isEqualTo("agent-published");
        verify(dictMapper, never()).selectOne(any(Wrapper.class));
    }

    @Test
    @DisplayName("Agent 级：无已发布版本回退主表可编辑态模型")
    void agentWithoutPublishedFallsBackToMainTable() {
        when(preferenceMapper.selectOne(any(Wrapper.class))).thenReturn(null);
        when(agentMapper.selectOne(any(Wrapper.class))).thenReturn(agent("t-agent", "agent-main"));
        when(agentVersionMapper.selectOne(any(Wrapper.class))).thenReturn(null);

        assertThat(resolver.resolveForUser(USER_ID, "t-agent")).isEqualTo("agent-main");
    }

    @Test
    @DisplayName("Agent 级：用户偏好优先于 Agent 模型")
    void userPreferenceBeatsAgentModel() {
        stubPreference("qwen3-0.6b");
        when(modelMapper.selectOne(any(Wrapper.class))).thenReturn(chatModel("qwen3-0.6b", 0));
        when(agentMapper.selectOne(any(Wrapper.class))).thenReturn(agent("t-agent", "agent-main"));

        assertThat(resolver.resolveForUser(USER_ID, "t-agent")).isEqualTo("qwen3-0.6b");
        verify(agentMapper, never()).selectOne(any(Wrapper.class));
    }

    @Test
    @DisplayName("Agent 级：Agent 不存在时回落平台默认")
    void unknownAgentFallsBackToPlatformDefault() {
        when(preferenceMapper.selectOne(any(Wrapper.class))).thenReturn(null);
        when(agentMapper.selectOne(any(Wrapper.class))).thenReturn(null);
        when(dictMapper.selectOne(any(Wrapper.class))).thenReturn(platformDefault("qwen3-0.6b"));

        assertThat(resolver.resolveForUser(USER_ID, "ghost-agent")).isEqualTo("qwen3-0.6b");
    }

    @Test
    @DisplayName("设为偏好：模型不存在/未启用/非 chat（查询无命中）报 A0601")
    void assertSelectableRejectsUnavailableModel() {
        when(modelMapper.selectOne(any(Wrapper.class))).thenReturn(null);

        BusinessException ex =
                assertThrows(BusinessException.class, () -> resolver.assertSelectable("ghost-model", USER_ID));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.AI_MODEL_NOT_AVAILABLE);
        assertThat(ex.getMessage()).isEqualTo("模型不存在、未启用或当前等级不可用");
    }

    @Test
    @DisplayName("设为偏好：模型 vip_level 高于用户会员等级报 A0601（等级与启用模型列表同源）")
    void assertSelectableRejectsModelAboveUserLevel() {
        when(modelMapper.selectOne(any(Wrapper.class))).thenReturn(chatModel("gpt-4o", 2));
        when(userLevelResolver.levelOf(USER_ID)).thenReturn(1);

        BusinessException ex =
                assertThrows(BusinessException.class, () -> resolver.assertSelectable("gpt-4o", USER_ID));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.AI_MODEL_NOT_AVAILABLE);
        assertThat(ex.getMessage()).isEqualTo("模型不存在、未启用或当前等级不可用");
    }

    @Test
    @DisplayName("设为偏好：vip_level 为空按 0 处理（所有等级可用）")
    void assertSelectableAcceptsBlankVipLevel() {
        when(modelMapper.selectOne(any(Wrapper.class))).thenReturn(chatModel("qwen3-0.6b", null));
        when(userLevelResolver.levelOf(USER_ID)).thenReturn(0);

        resolver.assertSelectable("qwen3-0.6b", USER_ID);
    }

    @Test
    @DisplayName("设为偏好：等级不低于 vip_level 时放行")
    void assertSelectableAcceptsWithinUserLevel() {
        when(modelMapper.selectOne(any(Wrapper.class))).thenReturn(chatModel("level-1-model", 1));
        when(userLevelResolver.levelOf(USER_ID)).thenReturn(1);

        resolver.assertSelectable("level-1-model", USER_ID);
    }

    @Test
    @DisplayName("平台默认字典查询限定 type_code/name/status（禁用字典项不参与解析）")
    void platformDefaultLookupEnforcesEnabledDict() {
        when(preferenceMapper.selectOne(any(Wrapper.class))).thenReturn(null);
        when(dictMapper.selectOne(any(Wrapper.class))).thenReturn(platformDefault("qwen3-0.6b"));

        resolver.resolveForUser(USER_ID, null);

        ArgumentCaptor<Wrapper<SysDict>> captor = ArgumentCaptor.forClass(Wrapper.class);
        verify(dictMapper).selectOne(captor.capture());
        AbstractWrapper<?, ?, ?> wrapper = (AbstractWrapper<?, ?, ?>) captor.getValue();
        assertThat(wrapper.getSqlSegment())
                .contains("type_code")
                .contains("name")
                .contains("status");
        assertThat(new ArrayList<>(wrapper.getParamNameValuePairs().values())).contains("ai_model", "default_model", 1);
    }
}

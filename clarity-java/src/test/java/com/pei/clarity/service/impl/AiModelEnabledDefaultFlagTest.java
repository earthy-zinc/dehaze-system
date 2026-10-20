package com.pei.clarity.service.impl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.clarity.mapper.SysAiModelMapper;
import com.pei.clarity.mapper.SysAiModelPriceDetailMapper;
import com.pei.clarity.mapper.SysAiModelPriceMapper;
import com.pei.clarity.model.vo.AiModelVO;
import com.pei.clarity.security.util.SecurityUtils;
import com.pei.clarity.service.AiDefaultModelResolver;
import com.pei.clarity.service.AiProviderHealthService;
import com.pei.clarity.service.AiUserLevelResolver;
import com.pei.clarity.service.FavoriteService;
import com.pei.clarity.service.MessageService;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

/**
 * 启用模型列表的平台默认标记（isDefault）契约。
 *
 * <p>事实源为 python {@code AiModelService.list_enabled_models}：按 model_type 分键读字典
 * （chat → {@code ai_model.default_model}，其余 → {@code default_model.{modelType}}），
 * 命中平台默认的模型行 isDefault=true，其余 false；标记在读取时计算，不写入共享缓存快照。
 * 前端（front-vue/front-react 图像页）在无历史记录时以该标记预选后端默认模型，
 * 替代"启用列表首项"兜底。
 */
@DisplayName("启用模型列表：平台默认标记（isDefault）")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiModelEnabledDefaultFlagTest {

    private static final String CACHE_KEY = "ai:model:list";

    /** 与 python `AiModelResult.model_dump(mode="json")` 同格式的共享缓存快照（snake_case） */
    private static final String CACHE_SNAPSHOT = "["
            + "{\"id\":1,\"provider_id\":1,\"model_id\":\"chat-a\",\"model_type\":\"chat\","
            + "\"display_name\":\"A\",\"status\":1,\"vip_level\":0},"
            + "{\"id\":2,\"provider_id\":1,\"model_id\":\"chat-b\",\"model_type\":\"chat\","
            + "\"display_name\":\"B\",\"status\":1,\"vip_level\":0},"
            + "{\"id\":3,\"provider_id\":1,\"model_id\":\"aecrnet\",\"model_type\":\"image_edit\","
            + "\"display_name\":\"AECRNet\",\"status\":1,\"vip_level\":0}]";

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
    private StringRedisTemplate redis;

    @Mock
    private ValueOperations<String, String> valueOperations;

    @Mock
    private AiDefaultModelResolver aiDefaultModelResolver;

    private AiModelServiceImpl service;

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
                new ObjectMapper(),
                aiDefaultModelResolver);
        when(redis.opsForValue()).thenReturn(valueOperations);
        when(valueOperations.get(CACHE_KEY)).thenReturn(CACHE_SNAPSHOT);
        when(userLevelResolver.levelOf(any())).thenReturn(0);
    }

    @Test
    @DisplayName("标记取值与字典一致：命中平台默认的模型行为 true，其余为 false")
    void marksDefaultModelOfEachType() {
        when(aiDefaultModelResolver.resolvePlatformDefaultModelId("chat")).thenReturn("chat-a");
        when(aiDefaultModelResolver.resolvePlatformDefaultModelId("image_edit")).thenReturn("aecrnet");

        List<AiModelVO> items = listEnabled();

        assertThat(items).extracting(AiModelVO::getModelId).containsExactly("chat-a", "chat-b", "aecrnet");
        assertThat(items)
                .extracting(AiModelVO::getModelId, AiModelVO::getIsDefault)
                .containsExactly(tuple("chat-a", true), tuple("chat-b", false), tuple("aecrnet", true));
        // 与字典键口径一致：chat 用 default_model、image_edit 用 default_model.image_edit
        assertThat(AiDefaultModelResolver.dictKeyFor("chat")).isEqualTo("default_model");
        assertThat(AiDefaultModelResolver.dictKeyFor("image_edit")).isEqualTo("default_model.image_edit");
    }

    @Test
    @DisplayName("按类型筛选时只读出现到的类型字典键，且不改变过滤结果")
    void readsDictOnlyForPresentTypes() {
        when(aiDefaultModelResolver.resolvePlatformDefaultModelId("image_edit")).thenReturn("aecrnet");

        List<AiModelVO> items = listEnabled("image_edit", null);

        assertThat(items).extracting(AiModelVO::getModelId).containsExactly("aecrnet");
        assertThat(items.get(0).getIsDefault()).isTrue();
        verify(aiDefaultModelResolver).resolvePlatformDefaultModelId(eq("image_edit"));
        verify(aiDefaultModelResolver, never()).resolvePlatformDefaultModelId(eq("chat"));
    }

    @Test
    @DisplayName("字典未配置该类型默认模型时全部为 false（不回落列表首项）")
    void unconfiguredDefaultYieldsNoFlag() {
        when(aiDefaultModelResolver.resolvePlatformDefaultModelId(any())).thenReturn(null);

        List<AiModelVO> items = listEnabled();

        assertThat(items).allSatisfy(item -> assertThat(item.getIsDefault()).isFalse());
    }

    private List<AiModelVO> listEnabled() {
        return listEnabled(null, null);
    }

    private List<AiModelVO> listEnabled(String modelType, String taskTags) {
        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(7L);
            return service.listEnabledModels(modelType, taskTags);
        }
    }
}

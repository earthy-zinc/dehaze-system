package com.pei.dehaze.service.impl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.baomidou.mybatisplus.core.toolkit.GlobalConfigUtils;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAiModelMapper;
import com.pei.dehaze.mapper.SysAiModelPriceDetailMapper;
import com.pei.dehaze.mapper.SysAiModelPriceMapper;
import com.pei.dehaze.model.entity.SysAiModel;
import com.pei.dehaze.service.AiDefaultModelResolver;
import com.pei.dehaze.service.AiProviderHealthService;
import com.pei.dehaze.service.AiUserLevelResolver;
import com.pei.dehaze.service.FavoriteService;
import com.pei.dehaze.service.MessageService;
import java.util.List;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InOrder;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.test.util.ReflectionTestUtils;

/**
 * 模型删除 → 收藏失效联动（SDK 用例 favorite.test.ts「完整联动：模型删除后收藏标记 isInvalid=true」）。
 *
 * <p>三端同口径：软删模型 → 标记该模型收藏失效（target_id = sys_ai_model.id）→ 清启用模型缓存；
 * 与 python {@code delete_model}、Java 数据集删除路径同序。
 */
@DisplayName("AiModelServiceImpl 模型删除联动收藏失效")
@ExtendWith(MockitoExtension.class)
class AiModelDeleteFavoriteInvalidationTest {

    private static final long MODEL_PK = 13L;
    private static final String MODEL_SLUG = "aecrnet";

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
    @DisplayName("删除模型：先软删、再按主键标记收藏失效、最后清缓存")
    void deleteModel_marksFavoritesInvalidAfterSoftDelete() {
        givenModel();
        when(modelMapper.countActiveConversations(MODEL_SLUG)).thenReturn(0L);
        when(modelMapper.selectCount(any())).thenReturn(0L);

        service.deleteModel(MODEL_SLUG);

        InOrder order = inOrder(modelMapper, favoriteService, redis);
        // MyBatis-Plus 逻辑删除按实体下发（id 承载在实体上）
        ArgumentCaptor<SysAiModel> deleted = ArgumentCaptor.forClass(SysAiModel.class);
        order.verify(modelMapper).deleteById(deleted.capture());
        assertThat(deleted.getValue().getId()).isEqualTo(MODEL_PK);
        order.verify(favoriteService).markInvalid("model", List.of(MODEL_PK));
        order.verify(redis).delete("ai:model:list");
    }

    @Test
    @DisplayName("删除不存在的模型：A0401 且不标记收藏、不清缓存")
    void deleteModel_missingModelSkipsInvalidation() {
        when(modelMapper.selectList(any())).thenReturn(List.of());

        assertThatThrownBy(() -> service.deleteModel("ghost-model"))
                .isInstanceOf(BusinessException.class)
                .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                .isEqualTo(ResultCode.RESOURCE_NOT_FOUND.getCode());

        verifyNoInteractions(favoriteService);
        verify(redis, never()).delete(anyString());
    }

    @Test
    @DisplayName("占用校验失败：A0500 且不软删、不标记收藏")
    void deleteModel_activeConversationSkipsInvalidation() {
        givenModel();
        when(modelMapper.countActiveConversations(MODEL_SLUG)).thenReturn(2L);

        assertThatThrownBy(() -> service.deleteModel(MODEL_SLUG))
                .isInstanceOf(BusinessException.class)
                .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                .isEqualTo(ResultCode.DATA_REFERENCED.getCode());

        verifyNoInteractions(favoriteService);
        verify(modelMapper, never()).deleteById(any(SysAiModel.class));
    }

    private void givenModel() {
        SysAiModel model = new SysAiModel();
        model.setId(MODEL_PK);
        model.setModelId(MODEL_SLUG);
        when(modelMapper.selectList(any())).thenReturn(List.of(model));
    }
}

package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.exception.GlobalExceptionHandler;
import com.pei.dehaze.mapper.SysEvalLogMapper;
import com.pei.dehaze.model.query.DatasetItemQuery;
import com.pei.dehaze.model.query.FavoritePageQuery;
import com.pei.dehaze.model.query.MessageQuery;
import com.pei.dehaze.model.query.MessageTemplateQuery;
import com.pei.dehaze.service.DatasetOperationService;
import com.pei.dehaze.service.FavoriteService;
import com.pei.dehaze.service.MessageService;
import com.pei.dehaze.service.MessageTemplateService;
import com.pei.dehaze.service.SysDatasetItemService;
import com.pei.dehaze.service.SysEvalLogService;
import com.pei.dehaze.service.SysInputHistoryService;
import com.pei.dehaze.service.SysPredLogService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.validation.beanvalidation.LocalValidatorFactoryBean;

import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * 系统域分页边界与 python 默认值对齐（三端 `pageSize ∈ [1,100]` 统一口径）。
 *
 * <p>两类缺口一次守住：
 * <ol>
 *   <li><b>死注解（A 类）</b>：`@Min(1)/@Max(100)` 写在 {@code BasePageQuery} 上，但 controller 参数缺
 *       {@code @Valid} 时完全不生效——越界值静默直通，只由 MyBatis-Plus `setMaxLimit(200)` 在
 *       {@code >200} 时静默改小，与 python/go 的 A0400 分叉；</li>
 *   <li><b>默认值漂移</b>：python 侧 favorite / message / message_template / dataset_item 的
 *       {@code pageSize} 默认是 <b>20</b>（非 `BasePageQuery` 的 10），java 原样继承父类 → 不传参时
 *       两端返回不同页长。此处用 {@code ArgumentCaptor} 断言控制器真实收到 20。</li>
 * </ol>
 *
 * <p>越界/下界断言覆盖 8 个端点（4 个默认 20 的端点 + 3 个自持 {@code pageSize} 的 B 类端点 +
 * 评测 logs），其余同型端点由类型同构与全量回归兜底（team-lead 认可的抽样口径）。
 * 安全起见不触达 service：400 在进入方法前返回，故未 mock `SecurityUtils`（`/evaluation/metrics`
 * 虽用 currentUser，但校验先于方法体执行）。
 */
@DisplayName("系统域分页边界（A0400）与默认值对齐")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class SysDomainPagingValidationTest {

    @Mock
    private FavoriteService favoriteService;
    @Mock
    private MessageService messageService;
    @Mock
    private MessageTemplateService messageTemplateService;
    @Mock
    private SysDatasetItemService sysDatasetItemService;
    @Mock
    private DatasetOperationService datasetOperationService;
    @Mock
    private SysEvalLogService evalLogService;
    @Mock
    private SysEvalLogMapper evalLogMapper;
    @Mock
    private SysInputHistoryService historyService;
    @Mock
    private SysPredLogService predLogService;

    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        LocalValidatorFactoryBean validator = new LocalValidatorFactoryBean();
        validator.afterPropertiesSet();
        mockMvc = MockMvcBuilders.standaloneSetup(
                        new FavoriteController(favoriteService),
                        new MessageController(messageService),
                        new MessageTemplateController(messageTemplateService),
                        new SysDatasetItemController(sysDatasetItemService, datasetOperationService),
                        new EvaluationController(evalLogService, evalLogMapper),
                        new ImageInputController(historyService),
                        new PredictionController(predLogService))
                .setControllerAdvice(new GlobalExceptionHandler())
                .setValidator(validator)
                .build();
    }

    static Stream<String> pagedEndpoints() {
        return Stream.of(
                "/api/v1/favorites/page",
                "/api/v1/messages",
                "/api/v1/message-templates/page",
                "/api/v1/dataset-items",
                "/api/v1/evaluation/metrics",
                "/api/v1/evaluation/logs",
                "/api/v1/image-input/history",
                "/api/v1/prediction/logs");
    }

    @ParameterizedTest(name = "{0}?pageSize=101 → 400")
    @MethodSource("pagedEndpoints")
    @DisplayName("pageSize=101 → 400 + A0400（补 @Valid 后死注解已激活）")
    void pageSizeOverLimitRejected(String url) throws Exception {
        mockMvc.perform(get(url).param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @ParameterizedTest(name = "{0}?pageNum=0 → 400")
    @MethodSource("pagedEndpoints")
    @DisplayName("pageNum=0 → 400 + A0400（下界，对齐 python ge=1）")
    void pageNumBelowMinRejected(String url) throws Exception {
        mockMvc.perform(get(url).param("pageNum", "0"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @Test
    @DisplayName("favorite 不传 pageSize → 20（python favorite.py:26，非父类默认 10）")
    void favoriteDefaultsToTwenty() throws Exception {
        when(favoriteService.getPage(any())).thenReturn(new Page<>());

        mockMvc.perform(get("/api/v1/favorites/page")).andExpect(status().isOk());

        ArgumentCaptor<FavoritePageQuery> captor = ArgumentCaptor.forClass(FavoritePageQuery.class);
        verify(favoriteService).getPage(captor.capture());
        assertThat(captor.getValue().getPageSize()).isEqualTo(20);
    }

    @Test
    @DisplayName("message 不传 pageSize → 20（python message.py:23）")
    void messageDefaultsToTwenty() throws Exception {
        when(messageService.getPage(any())).thenReturn(new Page<>());

        mockMvc.perform(get("/api/v1/messages")).andExpect(status().isOk());

        ArgumentCaptor<MessageQuery> captor = ArgumentCaptor.forClass(MessageQuery.class);
        verify(messageService).getPage(captor.capture());
        assertThat(captor.getValue().getPageSize()).isEqualTo(20);
    }

    @Test
    @DisplayName("messageTemplate 不传 pageSize → 20（python message_template.py:21）")
    void messageTemplateDefaultsToTwenty() throws Exception {
        when(messageTemplateService.getPage(any())).thenReturn(new Page<>());

        mockMvc.perform(get("/api/v1/message-templates/page")).andExpect(status().isOk());

        ArgumentCaptor<MessageTemplateQuery> captor = ArgumentCaptor.forClass(MessageTemplateQuery.class);
        verify(messageTemplateService).getPage(captor.capture());
        assertThat(captor.getValue().getPageSize()).isEqualTo(20);
    }

    @Test
    @DisplayName("datasetItem 不传 pageSize → 20（python dataset_item.py:41）")
    void datasetItemDefaultsToTwenty() throws Exception {
        when(sysDatasetItemService.pageSearchDatasetItems(any())).thenReturn(new Page<>());

        mockMvc.perform(get("/api/v1/dataset-items")).andExpect(status().isOk());

        ArgumentCaptor<DatasetItemQuery> captor = ArgumentCaptor.forClass(DatasetItemQuery.class);
        verify(sysDatasetItemService).pageSearchDatasetItems(captor.capture());
        assertThat(captor.getValue().getPageSize()).isEqualTo(20);
    }

    @Test
    @DisplayName("边界放行：pageSize=100 通过校验并原样透传，不误拦")
    void pageSizeAtLimitAccepted() throws Exception {
        when(favoriteService.getPage(any())).thenReturn(new Page<>());

        mockMvc.perform(get("/api/v1/favorites/page").param("pageSize", "100"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));

        ArgumentCaptor<FavoritePageQuery> captor = ArgumentCaptor.forClass(FavoritePageQuery.class);
        verify(favoriteService).getPage(captor.capture());
        assertThat(captor.getValue().getPageSize()).isEqualTo(100);
    }
}

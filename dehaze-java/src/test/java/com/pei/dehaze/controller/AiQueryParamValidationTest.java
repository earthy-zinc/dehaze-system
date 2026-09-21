package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.exception.GlobalExceptionHandler;
import com.pei.dehaze.model.query.AiBillingStatQuery;
import com.pei.dehaze.model.query.AiObservabilityTrendsQuery;
import com.pei.dehaze.service.AiBillingService;
import com.pei.dehaze.service.AiCompatCallService;
import com.pei.dehaze.service.AiObservabilityService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.validation.beanvalidation.LocalValidatorFactoryBean;

import java.util.List;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.verify;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * AI 域查询参数校验守卫。
 *
 * <p>教训来自 java-proxy 的取证：DTO 上写了 {@code @Min/@Max} 但控制器参数缺 {@code @Valid}
 * 时注解是「死注解」——绑定不触发校验，非法分页会被静默接受。故此处不测 DTO 注解本身，
 * 而是经真实 MVC 绑定链路（standalone MockMvc + 项目 GlobalExceptionHandler）断言
 * {@code pageSize/size} 越界返回 HTTP 400 + A0400，同时断言边界值 100 仍放行，避免过度拦截。
 */
@DisplayName("AI 域查询参数校验（@ParameterObject + @Valid 真实生效）")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiQueryParamValidationTest {

    @Mock
    private AiBillingService billingService;

    @Mock
    private AiObservabilityService observabilityService;

    @Mock
    private AiCompatCallService compatCallService;

    private MockMvc mockMvc;

    /**
     * 我域全部分页查询端点（路径 + 分页参数名）：compat 审计按 python 口径用 size，其余用 pageSize
     */
    static Stream<Arguments> pagedEndpoints() {
        return Stream.of(
                Arguments.of("/api/v1/ai-billing/records", "pageSize"),
                Arguments.of("/api/v1/ai-billing/credit-logs", "pageSize"),
                Arguments.of("/api/v1/ai-billing/refunds", "pageSize"),
                Arguments.of("/api/v1/ai-billing/anomalies", "pageSize"),
                Arguments.of("/api/v1/ai-billing/costs", "pageSize"),
                Arguments.of("/api/v1/ai/observability/traces", "pageSize"),
                Arguments.of("/api/v1/ai/observability/traces/export", "pageSize"),
                Arguments.of("/api/v1/ai/observability/costs", "pageSize"),
                Arguments.of("/api/v1/ai/compat/calls", "size"));
    }

    @BeforeEach
    void setUp() {
        LocalValidatorFactoryBean validator = new LocalValidatorFactoryBean();
        validator.afterPropertiesSet();
        mockMvc = MockMvcBuilders.standaloneSetup(
                        new AiBillingController(billingService),
                        new AiObservabilityController(observabilityService),
                        new AiCompatCallController(compatCallService))
                .setControllerAdvice(new GlobalExceptionHandler())
                .setValidator(validator)
                .build();

        when(billingService.listRecords(any())).thenReturn(new Page<>(1, 20, 0));
        when(billingService.listCreditLogs(any())).thenReturn(new Page<>(1, 20, 0));
        when(billingService.listRefunds(any())).thenReturn(new Page<>(1, 20, 0));
        when(billingService.listAnomalies(any())).thenReturn(new Page<>(1, 20, 0));
        when(billingService.listCosts(any())).thenReturn(new Page<>(1, 20, 0));
        when(billingService.getStats(any())).thenReturn(List.of());
        when(billingService.getBalance(any())).thenReturn(new com.pei.dehaze.model.vo.AiBalanceVO());
        when(observabilityService.getTimeline(any(), any()))
                .thenReturn(new com.pei.dehaze.model.vo.AiObservabilityTimelineVO());
        when(observabilityService.listTraces(any())).thenReturn(new Page<>(1, 10, 0));
        when(observabilityService.exportTraces(any())).thenReturn(new byte[0]);
        when(observabilityService.costs(any())).thenReturn(new com.pei.dehaze.model.vo.AiObservabilityCostsVO());
        when(observabilityService.trends(any())).thenReturn(List.of());
        when(compatCallService.listCalls(any())).thenReturn(new Page<>(1, 20, 0));
    }

    @ParameterizedTest(name = "{0} 的 {1}=101 触发校验并返回 A0400")
    @MethodSource("pagedEndpoints")
    @DisplayName("分页参数超上限：注解真实触发校验（非死注解）")
    void pageSizeOverLimitRejected(String url, String param) throws Exception {
        mockMvc.perform(get(url).param(param, "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @ParameterizedTest(name = "{0} 的 {1}=100 通过校验（边界放行）")
    @MethodSource("pagedEndpoints")
    @DisplayName("分页参数等于上限：放行，不误拦")
    void pageSizeAtLimitAccepted(String url, String param) throws Exception {
        var actions = mockMvc.perform(get(url).param(param, "100"))
                .andExpect(status().isOk());
        // 导出端点返回文件流而非 {code,msg,data} 信封，只断言状态码
        if (!url.endsWith("/export")) {
            actions.andExpect(jsonPath("$.code").value("00000"));
        }
    }

    @ParameterizedTest(name = "{0} 的 {1}=0 触发下限校验")
    @MethodSource("pagedEndpoints")
    @DisplayName("分页参数低于下限（0）同样 A0400")
    void pageSizeBelowLimitRejected(String url, String param) throws Exception {
        mockMvc.perform(get(url).param(param, "0"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @Test
    @DisplayName("页码越界（pageNum=0）同样被拦截")
    void pageNumBelowLimitRejected() throws Exception {
        mockMvc.perform(get("/api/v1/ai-billing/records").param("pageNum", "0"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @Test
    @DisplayName("过程链检索的 capability 非法由服务层拦截（A0400），参数校验不误伤合法取值")
    void capabilityValidationStaysInServiceLayer() throws Exception {
        when(observabilityService.listTraces(any()))
                .thenThrow(new com.pei.dehaze.common.exception.BusinessException(
                        com.pei.dehaze.common.result.ResultCode.PARAM_ERROR, "capability 仅支持 memory/kb/tools"));

        mockMvc.perform(get("/api/v1/ai/observability/traces").param("capability", "voice"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @Test
    @DisplayName("管理员统计：非法 groupBy 经控制器→服务层白名单返回 A0400（HTTP 层可达）")
    void statsRejectsIllegalGroupByThroughMvc() throws Exception {
        when(billingService.getStats(any()))
                .thenThrow(new com.pei.dehaze.common.exception.BusinessException(
                        com.pei.dehaze.common.result.ResultCode.PARAM_ERROR, "groupBy 仅支持 user/model/billType/day"));

        mockMvc.perform(get("/api/v1/ai-billing/stats").param("groupBy", "week"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"))
                .andExpect(jsonPath("$.msg").value("groupBy 仅支持 user/model/billType/day"));
    }

    @Test
    @DisplayName("管理员统计：合法 groupBy 放行且参数原样透传到服务层")
    void statsAcceptsLegalGroupBy() throws Exception {
        when(billingService.getStats(any())).thenReturn(List.of());

        mockMvc.perform(get("/api/v1/ai-billing/stats").param("groupBy", "day"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));

        ArgumentCaptor<AiBillingStatQuery> captor = ArgumentCaptor.forClass(AiBillingStatQuery.class);
        verify(billingService).getStats(captor.capture());
        assertThat(captor.getValue().getGroupBy()).isEqualTo("day");
    }

    @Test
    @DisplayName("性能趋势：非法 dimension 经控制器→服务层白名单返回 A0400（HTTP 层可达）")
    void trendsRejectsIllegalDimensionThroughMvc() throws Exception {
        when(observabilityService.trends(any()))
                .thenThrow(new com.pei.dehaze.common.exception.BusinessException(
                        com.pei.dehaze.common.result.ResultCode.PARAM_ERROR, "dimension 仅支持 model/agent"));

        mockMvc.perform(get("/api/v1/ai/observability/trends").param("dimension", "user"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"))
                .andExpect(jsonPath("$.msg").value("dimension 仅支持 model/agent"));
    }

    @Test
    @DisplayName("性能趋势：合法 dimension 放行且参数原样透传到服务层")
    void trendsAcceptsLegalDimension() throws Exception {
        when(observabilityService.trends(any())).thenReturn(List.of());

        mockMvc.perform(get("/api/v1/ai/observability/trends").param("dimension", "agent"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));

        ArgumentCaptor<AiObservabilityTrendsQuery> captor =
                ArgumentCaptor.forClass(AiObservabilityTrendsQuery.class);
        verify(observabilityService).trends(captor.capture());
        assertThat(captor.getValue().getDimension()).isEqualTo("agent");
    }

    /** python 侧 userId 带 ge=1 的端点（`ai_billing.py:93/111/138/195`） */
    static Stream<Arguments> userIdBoundedEndpoints() {
        return Stream.of(
                Arguments.of("/api/v1/ai-billing/balance"),
                Arguments.of("/api/v1/ai-billing/records"),
                Arguments.of("/api/v1/ai-billing/credit-logs"),
                Arguments.of("/api/v1/ai-billing/refunds"));
    }

    @ParameterizedTest(name = "{0} 的 userId=0 被拒（python ge=1 对齐）")
    @MethodSource("userIdBoundedEndpoints")
    @DisplayName("跨端 parity：python 有界 userId 在 java 侧同样拦截 0")
    void userIdBelowLowerBoundRejected(String url) throws Exception {
        mockMvc.perform(get(url).param("userId", "0"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @ParameterizedTest(name = "{0} 的 userId=1 放行")
    @MethodSource("userIdBoundedEndpoints")
    @DisplayName("跨端 parity：userId 合法下界 1 正常放行")
    void userIdAtLowerBoundAccepted(String url) throws Exception {
        mockMvc.perform(get(url).param("userId", "1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }

    @Test
    @DisplayName("stats/anomalies 的 userId 保持无界（python 本身无 ge，不得反向加界）")
    void statsAndAnomaliesUserIdStayUnbounded() throws Exception {
        mockMvc.perform(get("/api/v1/ai-billing/stats").param("userId", "0"))
                .andExpect(status().isOk());
        when(billingService.listAnomalies(any())).thenReturn(new Page<>(1, 20, 0));
        mockMvc.perform(get("/api/v1/ai-billing/anomalies").param("userId", "0"))
                .andExpect(status().isOk());
    }

    /** python `TracePageQuery` 长度界：agentCode/model/keyword max_length=64，errorType max_length=32 */
    static Stream<Arguments> traceLengthBoundedParams() {
        return Stream.of(
                Arguments.of("agentCode", 64),
                Arguments.of("model", 64),
                Arguments.of("keyword", 64),
                Arguments.of("errorType", 32));
    }

    @ParameterizedTest(name = "过程链检索 {0} 超 {1} 字符被拒")
    @MethodSource("traceLengthBoundedParams")
    @DisplayName("跨端 parity：python max_length 在 java 侧同样拦截")
    void traceParamsOverLengthRejected(String param, int maxLength) throws Exception {
        mockMvc.perform(get("/api/v1/ai/observability/traces").param(param, "x".repeat(maxLength + 1)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @ParameterizedTest(name = "过程链检索 {0} 等于 {1} 字符放行")
    @MethodSource("traceLengthBoundedParams")
    @DisplayName("跨端 parity：长度边界值放行，不误拦")
    void traceParamsAtLengthAccepted(String param, int maxLength) throws Exception {
        mockMvc.perform(get("/api/v1/ai/observability/traces").param(param, "x".repeat(maxLength)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }

    @Test
    @DisplayName("会话时间线 include 超 32 字符被拒（python TimelineQuery max_length=32）")
    void timelineIncludeOverLengthRejected() throws Exception {
        mockMvc.perform(get("/api/v1/ai/observability/conversations/1/timeline")
                        .param("include", "x".repeat(33)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        mockMvc.perform(get("/api/v1/ai/observability/conversations/1/timeline").param("include", "raw"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }

    @Test
    @DisplayName("无分页参数时使用 DTO 默认值，不触发校验")
    void defaultPagingAccepted() throws Exception {
        List<String> urls = List.of("/api/v1/ai-billing/records", "/api/v1/ai/observability/traces",
                "/api/v1/ai/compat/calls");
        for (String url : urls) {
            mockMvc.perform(get(url))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.code").value("00000"));
        }
    }
}

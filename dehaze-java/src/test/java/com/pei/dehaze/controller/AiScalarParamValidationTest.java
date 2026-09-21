package com.pei.dehaze.controller;

import com.pei.dehaze.common.exception.GlobalExceptionHandler;
import com.pei.dehaze.model.vo.AiEvalReviewQueueVO;
import com.pei.dehaze.model.vo.AiNextTimesVO;
import com.pei.dehaze.service.AiEvalCenterService;
import com.pei.dehaze.service.AiScheduleService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.validation.beanvalidation.MethodValidationPostProcessor;

import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * AI 域标量查询参数校验守卫（非分页）。
 *
 * <p>分页参数走 {@code BasePageQuery} 派生 DTO + 参数级 {@code @Valid}，已由
 * {@link AiPaginationValidationTest} 覆盖；本类针对**裸 {@code @RequestParam} 标量**
 * （limit / status / count）——这类参数在 py 侧同样有上下界（{@code Query(ge=, le=)}），
 * java 侧此前只靠类型转换兜住格式错误，越界值会被静默接受并直达服务层：
 * limit=100000 会让趋势查询拉全量、count=10000 会让 Cron 预览空转。
 *
 * <p>校验载体是类级 {@code @Validated} + 参数注解（方法级校验）。生产环境由 Spring Boot
 * 的 {@code ValidationAutoConfiguration} 注册 {@code MethodValidationPostProcessor} 生效；
 * standalone MockMvc 不会自动装配该后置处理器，故在此显式包装，使测试断言的是
 * **真实的方法级校验链路**而非"注解存在"。
 *
 * <p>每个越界断言都配有边界内放行断言（1/500、1/2、1/20），防止过度收紧：
 * 只有"越界拒绝 + 边界放行"成对成立，才说明约束区间与 python 一致。
 */
@DisplayName("AI 域标量查询参数校验（@Validated 方法级真实生效）")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiScalarParamValidationTest {

    private static final String TRENDS = "/api/v1/ai/eval-center/trends";
    private static final String REVIEWS = "/api/v1/ai/eval-center/reviews";
    private static final String NEXT_TIMES = "/api/v1/ai/scheduled-tasks/next-times";

    @Mock
    private AiEvalCenterService evalCenterService;

    @Mock
    private AiScheduleService scheduleService;

    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        when(evalCenterService.trends(any(), any(), any(), anyInt())).thenReturn(List.of());
        when(evalCenterService.listReviews(any())).thenReturn(new AiEvalReviewQueueVO());
        when(scheduleService.previewNextTimes(any(), anyInt())).thenReturn(new AiNextTimesVO());

        MethodValidationPostProcessor methodValidation = new MethodValidationPostProcessor();
        methodValidation.afterPropertiesSet();

        mockMvc = MockMvcBuilders.standaloneSetup(
                        proxy(new AiEvalCenterController(evalCenterService), methodValidation),
                        proxy(new AiScheduleController(scheduleService), methodValidation))
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();
    }

    /** 模拟 Boot 的 ValidationAutoConfiguration：@Validated 类需经后置处理器代理才有方法级校验 */
    private static Object proxy(Object controller, MethodValidationPostProcessor processor) {
        return processor.postProcessAfterInitialization(controller, controller.getClass().getSimpleName());
    }

    @ParameterizedTest(name = "trends limit={0} → 400 + A0400")
    @ValueSource(strings = {"0", "501", "-1"})
    @DisplayName("趋势条数上限越界（对齐 python Query(100, ge=1, le=500)）")
    void trendsLimitOutOfRangeRejected(String limit) throws Exception {
        mockMvc.perform(get(TRENDS).param("limit", limit))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @ParameterizedTest(name = "trends limit={0} → 放行")
    @ValueSource(strings = {"1", "100", "500"})
    @DisplayName("趋势条数上限边界内放行（1/100/500 不被误拦）")
    void trendsLimitInRangeAccepted(String limit) throws Exception {
        mockMvc.perform(get(TRENDS).param("limit", limit))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }

    @ParameterizedTest(name = "reviews status={0} → 400 + A0400")
    @ValueSource(strings = {"0", "3", "-1"})
    @DisplayName("复核状态越界（对齐 python Query(None, ge=1, le=2)）")
    void reviewsStatusOutOfRangeRejected(String status) throws Exception {
        mockMvc.perform(get(REVIEWS).param("status", status))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @ParameterizedTest(name = "reviews status={0} → 放行")
    @ValueSource(strings = {"1", "2"})
    @DisplayName("复核状态合法取值放行（1:待复核 2:已复核）")
    void reviewsStatusInRangeAccepted(String status) throws Exception {
        mockMvc.perform(get(REVIEWS).param("status", status))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }

    @Test
    @DisplayName("复核状态缺省（不传 status）放行")
    void reviewsStatusAbsentAccepted() throws Exception {
        mockMvc.perform(get(REVIEWS))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }

    @ParameterizedTest(name = "next-times count={0} → 400 + A0400")
    @ValueSource(strings = {"0", "21", "-1"})
    @DisplayName("触发预览次数越界（对齐 python Query(5, ge=1, le=20)）")
    void nextTimesCountOutOfRangeRejected(String count) throws Exception {
        mockMvc.perform(get(NEXT_TIMES).param("cron", "0 0 * * *").param("count", count))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @ParameterizedTest(name = "next-times count={0} → 放行")
    @ValueSource(strings = {"1", "5", "20"})
    @DisplayName("触发预览次数边界内放行（1/5/20 不被误拦）")
    void nextTimesCountInRangeAccepted(String count) throws Exception {
        mockMvc.perform(get(NEXT_TIMES).param("cron", "0 0 * * *").param("count", count))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }

    @Test
    @DisplayName("非数字 limit → 类型转换失败 400 + A0400（不是 500）")
    void nonNumericLimitRejected() throws Exception {
        mockMvc.perform(get(TRENDS).param("limit", "abc"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @Test
    @DisplayName("时间参数格式非法 → 400 + A0400（对齐 python datetime 强类型）")
    void invalidDateTimeRejected() throws Exception {
        mockMvc.perform(get(TRENDS).param("startTime", "abc"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @Test
    @DisplayName("时间参数空串 → 视为未传（宽松口径），不报错")
    void blankDateTimeTreatedAsAbsent() throws Exception {
        mockMvc.perform(get(TRENDS).param("startTime", "").param("endTime", ""))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }
}

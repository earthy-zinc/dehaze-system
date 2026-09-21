package com.pei.dehaze.controller;

import com.pei.dehaze.common.exception.GlobalExceptionHandler;
import com.pei.dehaze.service.SysAlgorithmService;
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
import org.springframework.validation.beanvalidation.LocalValidatorFactoryBean;
import org.springframework.validation.beanvalidation.MethodValidationPostProcessor;

import java.util.List;

import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * 算法统计报表 `days` 参数校验（方法级）。
 *
 * <p>原实现是**静默回退**：控制器裸 `@RequestParam(defaultValue="7") Integer days` 无下界，
 * service 用 `days != null && days > 0 ? days : 7` 把 0/负数悄悄当成 7 —— 用户传 `days=0` 会拿到
 * "看起来正常、其实不是他要的"报表，且三端只有 python 报错。现按"非法输入必须产生可观测错误码
 * 而非默认值"改为参数层拒绝（`@Validated` + `@Min(1)`），service 的三元兜底一并删除。
 *
 * <p>方法级校验需 `MethodValidationPostProcessor` 代理才生效——standalone MockMvc 直接 new 出来的
 * controller 没有该代理，照抄字段级 `@Valid` 的测法会**全绿但什么都没验**，故此处显式包装代理。
 */
@DisplayName("算法报表 days 参数校验（方法级）")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class SysAlgorithmMonitorStatsValidationTest {

    private static final String STATS_URL = "/api/v1/algorithms/1/monitor/stats";

    @Mock
    private SysAlgorithmService algorithmService;

    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        LocalValidatorFactoryBean validator = new LocalValidatorFactoryBean();
        validator.afterPropertiesSet();
        MethodValidationPostProcessor methodValidation = new MethodValidationPostProcessor();
        methodValidation.afterPropertiesSet();

        // 本类只覆盖 getMonitorStats 路径，其余依赖（版本/转换器/导入导出/模板）不会被触达，传 null 即可
        Object controller = methodValidation.postProcessAfterInitialization(
                new SysAlgorithmController(algorithmService, null, null, null, null), "sysAlgorithmController");

        mockMvc = MockMvcBuilders.standaloneSetup(controller)
                .setControllerAdvice(new GlobalExceptionHandler())
                .setValidator(validator)
                .build();
    }

    @ParameterizedTest(name = "days={0} → 400 + A0400 且未触达 service")
    @ValueSource(strings = {"0", "-1", "-100"})
    @DisplayName("days 非法（0/负数）→ 400 + A0400，不再静默回退 7")
    void daysBelowMinRejected(String days) throws Exception {
        mockMvc.perform(get(STATS_URL).param("days", days))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(algorithmService);
    }

    @Test
    @DisplayName("不传 days → 用注解默认值 7 并透传给 service")
    void daysAbsentUsesDefaultSeven() throws Exception {
        when(algorithmService.getMonitorStats(1L, 7)).thenReturn(List.of());

        mockMvc.perform(get(STATS_URL))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));

        verify(algorithmService).getMonitorStats(1L, 7);
    }

    @Test
    @DisplayName("合法 days 原样透传（days=30，不被兜底改写）")
    void validDaysPassedThrough() throws Exception {
        when(algorithmService.getMonitorStats(1L, 30)).thenReturn(List.of());

        mockMvc.perform(get(STATS_URL).param("days", "30"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));

        verify(algorithmService).getMonitorStats(1L, 30);
    }

    @Test
    @DisplayName("days 非数字 → 400 + A0400（类型转换失败，非 500）")
    void nonNumericDaysRejected() throws Exception {
        mockMvc.perform(get(STATS_URL).param("days", "abc"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }
}

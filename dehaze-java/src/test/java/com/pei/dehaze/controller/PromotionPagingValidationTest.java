package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.exception.GlobalExceptionHandler;
import com.pei.dehaze.service.PromotionService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.validation.beanvalidation.LocalValidatorFactoryBean;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * 促销活动分页参数边界（`pageSize ≤ 100`，三端统一 `le=100` 口径）。
 *
 * <p>`PromotionPageQuery` 继承 `BasePageQuery` 的 `@Max(100)`，但控制器参数此前缺 `@Valid`
 * → 注解是**死注解**，越界值只由 MyBatis-Plus 的 `setMaxLimit(200)` 静默 clamp，
 * 与 python / go 的 `A0400` 分叉（越界应显式拒绝，而非静默改小后照常返回）。
 * 本类经真实 MVC 绑定链路断言"注解 + 触发条件"整体生效，并配边界放行防过度收紧。
 */
@DisplayName("促销活动分页边界（A0400）")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class PromotionPagingValidationTest {

    @Mock
    private PromotionService promotionService;

    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        LocalValidatorFactoryBean validator = new LocalValidatorFactoryBean();
        validator.afterPropertiesSet();
        mockMvc = MockMvcBuilders.standaloneSetup(new PromotionController(promotionService))
                .setControllerAdvice(new GlobalExceptionHandler())
                .setValidator(validator)
                .build();
    }

    @Test
    @DisplayName("pageSize=101 → 400 + A0400，且未触达服务层")
    void pageSizeOverLimitRejected() throws Exception {
        mockMvc.perform(get("/api/v1/packages/promotions/page").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(promotionService);
    }

    @Test
    @DisplayName("pageNum=0 → 400 + A0400（页码下界）")
    void pageNumBelowMinRejected() throws Exception {
        mockMvc.perform(get("/api/v1/packages/promotions/page").param("pageNum", "0"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(promotionService);
    }

    @Test
    @DisplayName("pageSize=100 边界放行，不误拦")
    void pageSizeAtLimitAccepted() throws Exception {
        when(promotionService.getPage(any())).thenReturn(new Page<>());

        mockMvc.perform(get("/api/v1/packages/promotions/page").param("pageSize", "100"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }
}

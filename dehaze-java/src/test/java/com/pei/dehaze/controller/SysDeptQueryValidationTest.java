package com.pei.dehaze.controller;

import com.pei.dehaze.common.exception.GlobalExceptionHandler;
import com.pei.dehaze.service.SysDeptService;
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

import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * 部门列表 `status` 参数校验（对齐 python `router/dept.py:25` 的 `Query(default=None, ge=0, le=1)`）。
 *
 * <p>原先 `DeptQuery.status` 无任何约束、控制器参数也缺 `@Valid` —— 两侧叠加使 `status=5` 之类
 * 直接穿透成过滤条件（静默返回意外结果），与 python 的 A0400 分叉。本类断言"字段注解 + 触发条件"
 * 整体生效，并覆盖 `keywords`（python 侧是裸 `str | None`，**刻意不加**约束）以防过度收紧。
 */
@DisplayName("部门查询 status 校验（ge=0,le=1）")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class SysDeptQueryValidationTest {

    private static final String DEPTS_URL = "/api/v1/depts";

    @Mock
    private SysDeptService deptService;

    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        LocalValidatorFactoryBean validator = new LocalValidatorFactoryBean();
        validator.afterPropertiesSet();
        mockMvc = MockMvcBuilders.standaloneSetup(new SysDeptController(deptService))
                .setControllerAdvice(new GlobalExceptionHandler())
                .setValidator(validator)
                .build();
    }

    @ParameterizedTest(name = "status={0} → 400 + A0400 且未触达 service")
    @ValueSource(strings = {"2", "-1", "5"})
    @DisplayName("status 越界（2/-1/5）→ 400 + A0400")
    void statusOutOfRangeRejected(String status) throws Exception {
        mockMvc.perform(get(DEPTS_URL).param("status", status))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(deptService);
    }

    @ParameterizedTest(name = "status={0} → 放行")
    @ValueSource(strings = {"0", "1"})
    @DisplayName("status=0/1 边界放行（python ge=0,le=1 的两端）")
    void statusInRangeAccepted(String status) throws Exception {
        when(deptService.listDepartments(any())).thenReturn(List.of());

        mockMvc.perform(get(DEPTS_URL).param("status", status))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }

    @Test
    @DisplayName("不传 status 放行（python default=None）")
    void statusAbsentAccepted() throws Exception {
        when(deptService.listDepartments(any())).thenReturn(List.of());

        mockMvc.perform(get(DEPTS_URL))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }

    @Test
    @DisplayName("keywords 无约束（python 是裸 str | None），任意长度放行")
    void keywordsUnconstrained() throws Exception {
        when(deptService.listDepartments(any())).thenReturn(List.of());

        mockMvc.perform(get(DEPTS_URL).param("keywords", "运维支撑部门（含超长说明".repeat(5)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }
}

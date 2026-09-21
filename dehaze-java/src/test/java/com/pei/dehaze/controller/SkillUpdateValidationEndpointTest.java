package com.pei.dehaze.controller;

import com.pei.dehaze.common.exception.GlobalExceptionHandler;
import com.pei.dehaze.model.form.SkillUpdateForm;
import com.pei.dehaze.model.vo.SkillVO;
import com.pei.dehaze.service.AiSkillService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.validation.beanvalidation.LocalValidatorFactoryBean;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * `PUT /api/v1/ai/skills/{id}` 的表单约束在端点层真实生效（`@RequestBody @Valid`）。
 * <p>
 * `SkillUpdateForm` 上的 `@Size` 若没有 `@Valid` 就是死注解——本类用 MockMvc 证明注解链路打通。
 */
@DisplayName("Skill 更新表单约束端点校验")
@ExtendWith(MockitoExtension.class)
class SkillUpdateValidationEndpointTest {

    @Mock
    private AiSkillService skillService;

    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        LocalValidatorFactoryBean validator = new LocalValidatorFactoryBean();
        validator.afterPropertiesSet();
        mockMvc = MockMvcBuilders.standaloneSetup(new AiSkillController(skillService))
                .setControllerAdvice(new GlobalExceptionHandler())
                .setValidator(validator)
                .build();
    }

    @Test
    @DisplayName("name 为空串 / 超长 返回 400 + A0400，且不触达服务层")
    void outOfRangeFieldsRejected() throws Exception {
        mockMvc.perform(put("/api/v1/ai/skills/5").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"name\":\"\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
        mockMvc.perform(put("/api/v1/ai/skills/5").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"name\":\"" + "x".repeat(129) + "\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
        mockMvc.perform(put("/api/v1/ai/skills/5").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"scene\":\"" + "x".repeat(256) + "\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(skillService);
    }

    @Test
    @DisplayName("边界内的更新请求放行并透传表单")
    void validUpdateAccepted() throws Exception {
        when(skillService.updateSkill(eq(5L), any(SkillUpdateForm.class))).thenReturn(new SkillVO());

        mockMvc.perform(put("/api/v1/ai/skills/5").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"name\":\"skill-a\",\"scene\":\"去雾\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));

        verify(skillService).updateSkill(eq(5L), any(SkillUpdateForm.class));
    }
}

package com.pei.dehaze.controller;

import com.pei.dehaze.common.exception.GlobalExceptionHandler;
import com.pei.dehaze.model.form.KbUpdateForm;
import com.pei.dehaze.model.vo.KbVO;
import com.pei.dehaze.service.AiKnowledgeBaseService;
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
 * `PUT /api/v1/kb/{id}` 的表单约束在端点层真实生效（`@RequestBody @Valid`）。
 * <p>
 * 字段级注解若没有 `@Valid` 就是死注解——本类用 MockMvc 证明注解链路打通，并钉住
 * python `score_threshold` 的 `lt=1` 语义（等于上界必须被拒，A0400）。
 */
@DisplayName("知识库编辑表单约束端点校验")
@ExtendWith(MockitoExtension.class)
class KbUpdateValidationEndpointTest {

    @Mock
    private AiKnowledgeBaseService knowledgeBaseService;

    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        LocalValidatorFactoryBean validator = new LocalValidatorFactoryBean();
        validator.afterPropertiesSet();
        mockMvc = MockMvcBuilders.standaloneSetup(new AiKnowledgeBaseController(knowledgeBaseService))
                .setControllerAdvice(new GlobalExceptionHandler())
                .setValidator(validator)
                .build();
    }

    @Test
    @DisplayName("scoreThreshold=1（等于上界，python 为 lt=1）返回 400 + A0400，且不触达服务层")
    void scoreThresholdAtUpperBoundRejected() throws Exception {
        mockMvc.perform(put("/api/v1/kb/9").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"scoreThreshold\":1}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(knowledgeBaseService);
    }

    @Test
    @DisplayName("topK=101 / hybridWeight=1.2 / rerankModel 超长 返回 400 + A0400")
    void otherOutOfRangeFieldsRejected() throws Exception {
        mockMvc.perform(put("/api/v1/kb/9").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"topK\":101}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
        mockMvc.perform(put("/api/v1/kb/9").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"hybridWeight\":1.2}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
        mockMvc.perform(put("/api/v1/kb/9").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"rerankModel\":\"" + "x".repeat(65) + "\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(knowledgeBaseService);
    }

    @Test
    @DisplayName("searchStrategy 非枚举值（python Literal[vector|keyword|hybrid]）返回 400 + A0400，且不触达服务层")
    void searchStrategyEnumRejected() throws Exception {
        mockMvc.perform(put("/api/v1/kb/9").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"searchStrategy\":\"bogus\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(knowledgeBaseService);
    }

    @Test
    @DisplayName("不可修改字段：chunkingStrategy 非法字面量拦在校验层(A0400)，合法值放行到服务层报不可修改")
    void immutableChunkingStrategyValidationOrder() throws Exception {
        mockMvc.perform(put("/api/v1/kb/9").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"chunkingStrategy\":\"bogus\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
        verifyNoInteractions(knowledgeBaseService);

        when(knowledgeBaseService.update(eq(9L), any(KbUpdateForm.class))).thenReturn(new KbVO());
        mockMvc.perform(put("/api/v1/kb/9").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"chunkingStrategy\":\"fixed\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
        verify(knowledgeBaseService).update(eq(9L), any(KbUpdateForm.class));
    }

    @Test
    @DisplayName("边界内的编辑请求放行并透传表单")
    void validUpdateAccepted() throws Exception {
        when(knowledgeBaseService.update(eq(9L), any(KbUpdateForm.class))).thenReturn(new KbVO());

        mockMvc.perform(put("/api/v1/kb/9").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"name\":\"知识库\",\"searchStrategy\":\"hybrid\",\"hybridWeight\":0.7,"
                                + "\"topK\":5,\"scoreThreshold\":0.5}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));

        verify(knowledgeBaseService).update(eq(9L), any(KbUpdateForm.class));
    }
}

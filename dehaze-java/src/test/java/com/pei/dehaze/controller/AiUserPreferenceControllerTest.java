package com.pei.dehaze.controller;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.pei.dehaze.common.exception.GlobalExceptionHandler;
import com.pei.dehaze.model.form.AiUserPreferenceForm;
import com.pei.dehaze.model.vo.AiUserPreferenceVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.AiUserPreferenceService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

/**
 * 用户偏好端点响应体口径：可空字段剔除后，未设置偏好时 {@code defaultModel} 键不出现
 * （python 全局 NonNullJSONResponse 为参照，java 靠 VO 上的 NON_NULL 对齐）。
 */
@DisplayName("AiUserPreferenceController 用户偏好端点响应口径")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiUserPreferenceControllerTest {

    private static final Long USER_ID = 5L;

    @Mock
    private AiUserPreferenceService userPreferenceService;

    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        mockMvc = MockMvcBuilders.standaloneSetup(new AiUserPreferenceController(userPreferenceService))
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();
    }

    @Test
    @DisplayName("GET：未设置偏好时响应体不含 defaultModel 键（非 null 值）")
    void getWithoutPreferenceOmitsDefaultModelKey() throws Exception {
        when(userPreferenceService.get(USER_ID)).thenReturn(new AiUserPreferenceVO());
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(USER_ID);

            mockMvc.perform(get("/api/v1/ai/user-preference"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.code").value("00000"))
                    .andExpect(jsonPath("$.data.defaultModel").doesNotExist());
        }
    }

    @Test
    @DisplayName("GET：已设置偏好时输出具体模型")
    void getWithPreferenceReturnsModel() throws Exception {
        AiUserPreferenceVO vo = new AiUserPreferenceVO();
        vo.setDefaultModel("qwen3-0.6b");
        when(userPreferenceService.get(USER_ID)).thenReturn(vo);
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(USER_ID);

            mockMvc.perform(get("/api/v1/ai/user-preference"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.data.defaultModel").value("qwen3-0.6b"));
        }
    }

    @Test
    @DisplayName("PUT：defaultModel=null 原样透传服务层（清除偏好）")
    void putNullClearsPreference() throws Exception {
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(USER_ID);

            mockMvc.perform(put("/api/v1/ai/user-preference")
                            .contentType(MediaType.APPLICATION_JSON)
                            .content("{\"defaultModel\":null}"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.code").value("00000"));
        }

        ArgumentCaptor<AiUserPreferenceForm> captor = ArgumentCaptor.forClass(AiUserPreferenceForm.class);
        verify(userPreferenceService).update(eq(USER_ID), captor.capture());
        assertThat(captor.getValue().getDefaultModel()).isNull();
    }
}

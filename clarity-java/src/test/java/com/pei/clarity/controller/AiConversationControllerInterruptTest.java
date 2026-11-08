package com.pei.clarity.controller;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.pei.clarity.common.exception.GlobalExceptionHandler;
import com.pei.clarity.model.dto.ai.AiContract;
import com.pei.clarity.security.util.SecurityUtils;
import com.pei.clarity.service.AiConversationService;
import java.util.Set;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

/**
 * 消息待恢复中断端点：{@code GET /api/v1/ai/messages/{msgId}/interrupt}。
 *
 * <p>无中断时 {@code data} 为 null（前端据此不展示确认条）；过期判定在服务层，端点只透传契约 {@code Interrupt}。
 */
@DisplayName("消息中断查询端点")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiConversationControllerInterruptTest {

    private static final String INTERRUPT = "/api/v1/ai/messages/9/interrupt";

    @Mock
    private AiConversationService conversationService;

    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        mockMvc = MockMvcBuilders.standaloneSetup(new AiConversationController(conversationService))
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();
    }

    @Test
    @DisplayName("无中断：200 + data=null，接口按当前用户透传消息 ID")
    void returnsNullWhenNoInterrupt() throws Exception {
        when(conversationService.getPendingInterrupt(9L, 1L, false)).thenReturn(null);

        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            mockMvc.perform(get(INTERRUPT))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.code").value("00000"))
                    .andExpect(jsonPath("$.data").doesNotExist());
        }
        verify(conversationService).getPendingInterrupt(eq(9L), eq(1L), eq(false));
    }

    @Test
    @DisplayName("有中断：返回契约形状（含 actionable=false 的不可续标注）")
    void returnsInterruptShape() throws Exception {
        AiContract.Interrupt expired = new AiContract.Interrupt(
                "int-1",
                "async_wait",
                null,
                "predict",
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                "约 30 秒",
                2L,
                null,
                null,
                "等待图像处理",
                "2026-10-21T10:00:00",
                "2026-10-21T10:01:00",
                false);
        when(conversationService.getPendingInterrupt(9L, 1L, false)).thenReturn(expired);

        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            mockMvc.perform(get(INTERRUPT))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.data.id").value("int-1"))
                    .andExpect(jsonPath("$.data.type").value("async_wait"))
                    .andExpect(jsonPath("$.data.actionable").value(false))
                    .andExpect(jsonPath("$.data.imageCount").value(2))
                    .andExpect(jsonPath("$.data.expiresAt").value("2026-10-21T10:01:00"));
        }
    }

    @Test
    @DisplayName("view=admin 无审计权限：403 + A0301，且不触达服务层")
    void adminViewRequiresAuditPermission() throws Exception {
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);
            security.when(SecurityUtils::isRoot).thenReturn(false);
            security.when(SecurityUtils::getPerms).thenReturn(Set.of());

            mockMvc.perform(get(INTERRUPT).param("view", "admin"))
                    .andExpect(status().isForbidden())
                    .andExpect(jsonPath("$.code").value("A0301"));
        }
        verifyNoInteractions(conversationService);
    }

    @Test
    @DisplayName("view=admin 持审计权限：以 admin 视角透传")
    void adminViewPropagatesFlag() throws Exception {
        when(conversationService.getPendingInterrupt(9L, 1L, true)).thenReturn(null);

        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);
            security.when(SecurityUtils::isRoot).thenReturn(false);
            security.when(SecurityUtils::getPerms).thenReturn(Set.of("ai:conversation:audit"));

            mockMvc.perform(get(INTERRUPT).param("view", "admin")).andExpect(status().isOk());
        }
        verify(conversationService).getPendingInterrupt(eq(9L), eq(1L), eq(true));
        verify(conversationService, org.mockito.Mockito.never()).getPendingInterrupt(eq(9L), isNull(), eq(false));
    }
}

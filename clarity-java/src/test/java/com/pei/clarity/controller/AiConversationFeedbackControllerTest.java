package com.pei.clarity.controller;

import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.pei.clarity.common.exception.GlobalExceptionHandler;
import com.pei.clarity.model.vo.AiFeedbackVO;
import com.pei.clarity.security.util.SecurityUtils;
import com.pei.clarity.service.AiFeedbackService;
import java.util.List;
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
 * 会话级批量反馈端点：{@code GET /api/v1/ai/conversations/{convId}/feedback}。
 *
 * <p>本人视角一次拉全会话内反馈，供前端进入会话时回填点赞/点踩状态。
 */
@DisplayName("会话级批量反馈端点")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiConversationFeedbackControllerTest {

    private static final String FEEDBACK = "/api/v1/ai/conversations/1/feedback";

    @Mock
    private AiFeedbackService feedbackService;

    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        mockMvc = MockMvcBuilders.standaloneSetup(new AiConversationFeedbackController(feedbackService))
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();
    }

    @Test
    @DisplayName("批量反馈：按当前用户透传会话 ID，返回 Feedback 数组")
    void listsFeedbackOfCurrentUser() throws Exception {
        AiFeedbackVO vo = new AiFeedbackVO();
        vo.setId(21L);
        vo.setMessageId(9L);
        vo.setUserId(1L);
        vo.setConversationId(1L);
        vo.setRating(1);
        vo.setTags(List.of("accurate"));
        when(feedbackService.listByConversation(1L, 1L)).thenReturn(List.of(vo));

        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            mockMvc.perform(get(FEEDBACK))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.code").value("00000"))
                    .andExpect(jsonPath("$.data[0].messageId").value(9))
                    .andExpect(jsonPath("$.data[0].rating").value(1))
                    .andExpect(jsonPath("$.data[0].tags[0]").value("accurate"));
        }
        verify(feedbackService).listByConversation(1L, 1L);
    }

    @Test
    @DisplayName("批量反馈：无反馈返回空数组")
    void returnsEmptyArrayWhenNoFeedback() throws Exception {
        when(feedbackService.listByConversation(1L, 1L)).thenReturn(List.of());

        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            mockMvc.perform(get(FEEDBACK))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.data").isArray())
                    .andExpect(jsonPath("$.data").isEmpty());
        }
    }
}

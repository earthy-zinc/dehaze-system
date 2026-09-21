package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.exception.GlobalExceptionHandler;
import com.pei.dehaze.model.query.AiConversationPageQuery;
import com.pei.dehaze.model.query.AiMessageCursorQuery;
import com.pei.dehaze.service.AiConversationService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.util.Arrays;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * 管理端会话审计视角（view=admin）鉴权单测。
 *
 * <p>口径对齐 dehaze-python {@code ai_conversation._require_conversation_audit}：ROOT 放行，否则需
 * {@code ai:conversation:audit}，越权抛 {@code ACCESS_UNAUTHORIZED}；该视角下会话列表/详情/消息列表/
 * 消息详情四处均必须校验，否则会横向越权读取他人会话。HTTP 状态码亦对齐 python 的 403。
 */
@DisplayName("会话审计视角鉴权（view=admin）")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiConversationControllerAdminViewTest {

    private static final String AUDIT_PERMISSION = "ai:conversation:audit";

    @Mock
    private AiConversationService conversationService;

    private AiConversationController controller;

    @BeforeEach
    void setUp() {
        controller = new AiConversationController(conversationService);
        SecurityContextHolder.clearContext();
        when(conversationService.list(any(), any(), anyBoolean()))
                .thenReturn(new Page<>(1, 10, 0));
    }

    @AfterEach
    void tearDown() {
        SecurityContextHolder.clearContext();
    }

    private AiConversationPageQuery adminQuery() {
        AiConversationPageQuery query = new AiConversationPageQuery();
        query.setView("admin");
        return query;
    }

    private void authenticate(String... authorities) {
        List<SimpleGrantedAuthority> granted = Arrays.stream(authorities)
                .map(SimpleGrantedAuthority::new).toList();
        SecurityContextHolder.getContext().setAuthentication(
                new UsernamePasswordAuthenticationToken("tester", "n/a", granted));
    }

    @Test
    @DisplayName("无审计权限请求 view=admin 抛 AccessDeniedException，且不触达服务层")
    void adminViewRejectedWithoutAuditPermission() {
        AccessDeniedException ex = assertThrows(AccessDeniedException.class, () -> controller.list(adminQuery()));

        assertThat(ex).isNotNull();
        verifyNoInteractions(conversationService);
    }

    @Test
    @DisplayName("越权访问经 GlobalExceptionHandler 返回 HTTP 403 + A0301 信封（与 python 403 口径一致）")
    void adminViewReturnsForbiddenEnvelope() throws Exception {
        MockMvc mockMvc = MockMvcBuilders.standaloneSetup(controller)
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();

        mockMvc.perform(get("/api/v1/ai/conversations").param("view", "admin"))
                .andExpect(status().isForbidden())
                .andExpect(jsonPath("$.code").value("A0301"))
                .andExpect(jsonPath("$.msg").value("访问未授权"));
    }

    @Test
    @DisplayName("持有 ai:conversation:audit 放行并以审计视角查询")
    void adminViewAllowedWithAuditPermission() {
        authenticate(AUDIT_PERMISSION);

        controller.list(adminQuery());

        verify(conversationService).list(isNull(), any(), eq(true));
    }

    @Test
    @DisplayName("ROOT 角色放行（无需显式审计权限）")
    void adminViewAllowedForRoot() {
        authenticate("ROLE_ROOT");

        controller.list(adminQuery());

        verify(conversationService).list(isNull(), any(), eq(true));
    }

    @Test
    @DisplayName("非 admin 视角不校验审计权限且按本人范围查询")
    void normalViewNeedsNoPermission() {
        controller.list(new AiConversationPageQuery());

        verify(conversationService).list(isNull(), any(), eq(false));
    }

    @Test
    @DisplayName("消息列表与消息详情同样受审计鉴权保护")
    void adminViewGuardedOnMessagesAndMessageDetail() {
        assertThrows(AccessDeniedException.class, () -> controller.listMessages(1L, new AiMessageCursorQuery(), "admin"));
        assertThrows(AccessDeniedException.class, () -> controller.getMessage(9L, "admin"));
        assertThrows(AccessDeniedException.class, () -> controller.detail(1L, "admin"));
    }
}

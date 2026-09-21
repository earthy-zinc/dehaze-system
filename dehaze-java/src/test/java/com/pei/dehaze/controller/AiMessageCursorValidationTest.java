package com.pei.dehaze.controller;

import com.pei.dehaze.common.exception.GlobalExceptionHandler;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.model.vo.AiMessagePageVO;
import com.pei.dehaze.model.vo.AiMessageVO;
import com.pei.dehaze.service.AiConversationService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.validation.beanvalidation.LocalValidatorFactoryBean;

import java.util.List;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * 会话消息列表游标分页守卫（契约：{@code before?}、{@code limit? 1..100 默认50}、响应 {@code {list,total,hasMore}}）。
 *
 * <p>校验载体是查询对象 {@code AiMessageCursorQuery}（字段级注解），由 Controller 参数上的 {@code @Valid}
 * 触发——因此断言的是**真实 MVC 绑定链路**而非"注解存在"：越界用例统一配 {@code verifyNoInteractions(service)}，
 * 证明请求在进入 Controller 方法前即被拦截，排除"进方法后业务兜底返回 A0400"的假阳性。
 *
 * <p>摘除反证：删掉 {@code AiMessageCursorQuery} 上的 {@code @Min/@Max}（或去掉参数级 {@code @Valid}）后，
 * 越界用例会因请求直达 service（返回 200 或 NPE）而转红。响应结构用例在移除 {@code hasMore} 装配后
 * （service 返回 null 字段）同样转红。
 *
 * <p>本类为字段级校验，使用 {@code setValidator} 装配真实校验器；未使用方法级
 * {@code MethodValidationPostProcessor}——消息端点无裸 {@code @RequestParam} 约束参数（{@code view} 无约束）。
 */
@DisplayName("会话消息游标分页校验（before/limit + 响应结构）")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiMessageCursorValidationTest {

    private static final String MESSAGES = "/api/v1/ai/conversations/1/messages";

    @Mock
    private AiConversationService conversationService;

    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        when(conversationService.listMessages(anyLong(), any(), any(), anyInt(), anyBoolean()))
                .thenReturn(pageVO(List.of(), 0L, false));

        LocalValidatorFactoryBean validator = new LocalValidatorFactoryBean();
        validator.afterPropertiesSet();
        mockMvc = MockMvcBuilders.standaloneSetup(new AiConversationController(conversationService))
                .setControllerAdvice(new GlobalExceptionHandler())
                .setValidator(validator)
                .build();
    }

    private static AiMessagePageVO pageVO(List<AiMessageVO> list, long total, boolean hasMore) {
        AiMessagePageVO vo = new AiMessagePageVO();
        vo.setList(list);
        vo.setTotal(total);
        vo.setHasMore(hasMore);
        return vo;
    }

    // ---------- before ----------

    @ParameterizedTest(name = "before={0} → 400 + A0400")
    @ValueSource(strings = {"0", "-1", "abc"})
    @DisplayName("before 低于下界/非数字被拒（对齐契约 ge=1）")
    void beforeOutOfRangeRejected(String before) throws Exception {
        mockMvc.perform(get(MESSAGES).param("before", before))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(conversationService);
    }

    @Test
    @DisplayName("before 边界=1 放行（下界本身合法）")
    void beforeAtMinAccepted() throws Exception {
        mockMvc.perform(get(MESSAGES).param("before", "1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }

    @Test
    @DisplayName("before 超出 Integer.MAX_VALUE：不被拒绝也不被截断，原值透传至服务层")
    void beforeAboveIntegerMaxPreservedUntruncated() throws Exception {
        // 消息主键为 BIGINT，游标必须按 64 位承载；若字段退回 Integer，绑定超大值会因溢出被拒（400）
        long bigBefore = (long) Integer.MAX_VALUE + 10L;

        mockMvc.perform(get(MESSAGES).param("before", Long.toString(bigBefore)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));

        verify(conversationService).listMessages(eq(1L), isNull(), eq(bigBefore), eq(50), eq(false));
    }

    @Test
    @DisplayName("before 缺省取最新一页：透传 before=null、limit=50")
    void beforeAbsentUsesLatestPage() throws Exception {
        mockMvc.perform(get(MESSAGES))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));

        verify(conversationService).listMessages(eq(1L), isNull(), isNull(), eq(50), eq(false));
    }

    // ---------- limit ----------

    @ParameterizedTest(name = "limit={0} → 400 + A0400")
    @ValueSource(strings = {"0", "101", "-1", "abc"})
    @DisplayName("limit 越界/非数字被拒（对齐契约 1..100）")
    void limitOutOfRangeRejected(String limit) throws Exception {
        mockMvc.perform(get(MESSAGES).param("limit", limit))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(conversationService);
    }

    @ParameterizedTest(name = "limit={0} → 放行")
    @ValueSource(strings = {"1", "100"})
    @DisplayName("limit 边界内放行（1/100 不被误拦）")
    void limitInRangeAccepted(String limit) throws Exception {
        mockMvc.perform(get(MESSAGES).param("limit", limit))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }

    @Test
    @DisplayName("limit 透传：显式值直达服务层")
    void limitPropagatesToService() throws Exception {
        mockMvc.perform(get(MESSAGES).param("limit", "100"))
                .andExpect(status().isOk());

        verify(conversationService).listMessages(eq(1L), isNull(), isNull(), eq(100), eq(false));
    }

    // ---------- 响应结构 ----------

    @Test
    @DisplayName("响应结构：data 暴露 list/total/hasMore（hasMore=true）")
    void responseExposesHasMoreAndTotal() throws Exception {
        AiMessageVO message = new AiMessageVO();
        message.setId(5L);
        when(conversationService.listMessages(anyLong(), any(), any(), anyInt(), anyBoolean()))
                .thenReturn(pageVO(List.of(message), 7L, true));

        mockMvc.perform(get(MESSAGES))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"))
                .andExpect(jsonPath("$.data.total").value(7))
                .andExpect(jsonPath("$.data.hasMore").value(true))
                .andExpect(jsonPath("$.data.list[0].id").value(5));
    }

    @Test
    @DisplayName("空会话：list 为空、total=0、hasMore=false")
    void emptyConversationShape() throws Exception {
        mockMvc.perform(get(MESSAGES))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.total").value(0))
                .andExpect(jsonPath("$.data.hasMore").value(false))
                .andExpect(jsonPath("$.data.list").isEmpty());
    }

    // ---------- view=admin ----------

    @Test
    @DisplayName("view=admin：需 ai:conversation:audit，并以审计视角透传 admin=true")
    void adminViewPropagatesFlag() throws Exception {
        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(1L);
            mocked.when(SecurityUtils::isRoot).thenReturn(false);
            mocked.when(SecurityUtils::getPerms).thenReturn(Set.of("ai:conversation:audit"));

            mockMvc.perform(get(MESSAGES).param("view", "admin"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.code").value("00000"));
        }

        verify(conversationService).listMessages(eq(1L), eq(1L), isNull(), eq(50), eq(true));
    }
}

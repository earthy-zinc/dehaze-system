package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.exception.GlobalExceptionHandler;
import com.pei.dehaze.service.AiAgentEndpointService;
import com.pei.dehaze.service.AiAgentService;
import com.pei.dehaze.service.AiAgentVersionService;
import com.pei.dehaze.service.AiArtifactService;
import com.pei.dehaze.service.AiConversationService;
import com.pei.dehaze.service.AiEvalService;
import com.pei.dehaze.service.AiMemoryService;
import com.pei.dehaze.service.AiScheduleService;
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
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * 分页参数边界校验（AI 域）：pageSize ≤ 100、pageNum ≥ 1，越界返回 400 + A0400。
 *
 * <p>口径对齐 dehaze-python {@code BasePageQuery}（{@code pageNum: ge=1}、{@code pageSize: ge=1, le=100}）：
 * 超限请求必须被拒而不是静默按超大页查询，否则一次 {@code pageSize=100000} 即可拖垮数据库。
 * 校验载体为 {@code BasePageQuery} 派生 DTO（无过滤条件的端点复用 {@code PageParamQuery}），
 * 由 Controller 上的 {@code @Valid} 触发；缺失 {@code @Valid} 时约束会退化为纯文档注解。
 *
 * <p>证据强度：越界用例额外断言 {@code verifyNoInteractions}，证明请求在进入 Controller 方法前
 * 即被 Bean Validation 拦截，而非"进入方法后由业务代码兜底返回 A0400"——后者即使缺少
 * 参数级 {@code @Valid} 也可能成立，无法证明注解真正生效。
 *
 * <p>覆盖范围：会话（列表/回收站）、Agent（列表/版本历史）、评测执行记录、记忆（列表/归档）、
 * 定时任务（列表/执行历史）、A2A 端点列表。会话产物列表由 {@code AiPagingParamValidationTest}
 * 覆盖（其覆盖更完整：含 {@code msg} 文案、边界放行、服务层透传与不触达断言），此处不重复注册。
 *
 * <p>会话消息列表已改为游标分页（{@code before}/{@code limit}），不再走 {@code PageParamQuery}，
 * 其参数校验由 {@code AiMessageCursorValidationTest} 覆盖。
 */
@DisplayName("AI 域分页边界校验（A0400）")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiPaginationValidationTest {

    @Mock
    private AiConversationService conversationService;
    @Mock
    private AiAgentService agentService;
    @Mock
    private AiAgentVersionService agentVersionService;
    @Mock
    private AiEvalService evalService;
    @Mock
    private AiMemoryService memoryService;
    @Mock
    private AiArtifactService artifactService;
    @Mock
    private AiScheduleService scheduleService;
    @Mock
    private AiAgentEndpointService endpointService;

    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        // 合法边界用例会触达服务层，统一返回空页
        when(conversationService.listTrash(any(), anyInt(), anyInt())).thenReturn(new Page<>(1, 100, 0));
        when(conversationService.list(any(), any(), anyBoolean())).thenReturn(new Page<>(1, 100, 0));
        when(agentVersionService.listVersions(anyLong(), anyInt(), anyInt())).thenReturn(new Page<>(1, 100, 0));
        when(evalService.listRuns(anyLong(), anyInt(), anyInt(), any())).thenReturn(new Page<>(1, 100, 0));
        when(memoryService.list(any(), anyInt(), anyInt(), any(), any())).thenReturn(new Page<>(1, 100, 0));
        when(memoryService.listArchived(any(), anyInt(), anyInt(), any())).thenReturn(new Page<>(1, 100, 0));
        when(scheduleService.listHistory(anyLong(), anyLong(), anyInt(), anyInt()))
                .thenReturn(new Page<>(1, 100, 0));

        // 显式注入校验器：standalone 默认 validator 依赖 classpath 装配，显式声明避免"注解失效"被误判为代码缺陷
        LocalValidatorFactoryBean validator = new LocalValidatorFactoryBean();
        validator.afterPropertiesSet();

        mockMvc = MockMvcBuilders.standaloneSetup(
                        new AiConversationController(conversationService),
                        new AiAgentController(agentService, agentVersionService,
                                new com.pei.dehaze.service.AiAgentConfigResolver(null)),
                        new AiAgentEvalController(evalService),
                        new AiMemoryController(memoryService),
                        new AiArtifactController(artifactService),
                        new AiScheduleController(scheduleService),
                        new AiAgentEndpointController(endpointService))
                .setControllerAdvice(new GlobalExceptionHandler())
                .setValidator(validator)
                .build();
    }

    /** 无过滤条件的端点：分页载体为 PageParamQuery */
    @Test
    @DisplayName("回收站列表：pageSize=101 返回 A0400，pageSize=100 放行")
    void trashPageSizeBoundary() throws Exception {
        mockMvc.perform(get("/api/v1/ai/conversations/trash").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
        mockMvc.perform(get("/api/v1/ai/conversations/trash").param("pageSize", "100"))
                .andExpect(status().isOk());
    }

    @Test
    @DisplayName("版本历史：pageSize=101 返回 A0400")
    void versionsPageSizeBoundary() throws Exception {
        mockMvc.perform(get("/api/v1/ai/agents/1/versions").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(agentVersionService);
    }

    @Test
    @DisplayName("定时任务执行历史：pageSize=101 返回 A0400")
    void scheduleHistoryPageSizeBoundary() throws Exception {
        mockMvc.perform(get("/api/v1/ai/scheduled-tasks/1/history").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(scheduleService);
    }

    @Test
    @DisplayName("评测执行记录：pageSize=101 返回 A0400（datasetId 与分页同 DTO 校验）")
    void evalRunsPageSizeBoundary() throws Exception {
        mockMvc.perform(get("/api/v1/ai/agents/1/eval/runs")
                        .param("pageSize", "101").param("datasetId", "5"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(evalService);
    }

    @Test
    @DisplayName("记忆列表：pageNum=0 返回 A0400（页码下界同样生效）")
    void memoryPageNumLowerBound() throws Exception {
        mockMvc.perform(get("/api/v1/ai/memories").param("pageNum", "0"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
        mockMvc.perform(get("/api/v1/ai/memories").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
        mockMvc.perform(get("/api/v1/ai/memories").param("pageSize", "100"))
                .andExpect(status().isOk());
    }

    @Test
    @DisplayName("归档记忆列表：pageSize=101 返回 A0400")
    void archivedMemoryPageSizeBoundary() throws Exception {
        mockMvc.perform(get("/api/v1/ai/memories/archived").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(memoryService);
    }

    @Test
    @DisplayName("Agent 列表与 A2A 端点列表：分页 DTO 校验同样生效")
    void otherPagedEndpointsBoundary() throws Exception {
        mockMvc.perform(get("/api/v1/ai/agents").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
        mockMvc.perform(get("/api/v1/ai/a2a/endpoints").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(agentService, endpointService);
    }

    @Test
    @DisplayName("会话列表：pageSize=101 返回 A0400（校验先于 view=admin 鉴权）")
    void conversationListPageSizeBoundary() throws Exception {
        mockMvc.perform(get("/api/v1/ai/conversations").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
        mockMvc.perform(get("/api/v1/ai/conversations").param("pageSize", "100"))
                .andExpect(status().isOk());
    }

    @Test
    @DisplayName("定时任务列表：pageSize=101 返回 A0400")
    void scheduleListPageSizeBoundary() throws Exception {
        mockMvc.perform(get("/api/v1/ai/scheduled-tasks").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(scheduleService);
    }
}

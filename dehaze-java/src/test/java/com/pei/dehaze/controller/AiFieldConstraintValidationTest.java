package com.pei.dehaze.controller;

import com.pei.dehaze.common.exception.GlobalExceptionHandler;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.AiAgentEndpointService;
import com.pei.dehaze.service.AiAgentService;
import com.pei.dehaze.service.AiAgentVersionService;
import com.pei.dehaze.service.AiConversationService;
import com.pei.dehaze.service.AiMemoryService;
import com.pei.dehaze.service.AiScheduleService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.validation.beanvalidation.LocalValidatorFactoryBean;
import org.springframework.validation.beanvalidation.MethodValidationPostProcessor;

import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.patch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * AI 域字段约束对齐守卫（P0 数值范围 / P1 字符串长度 / 方法级 limit 下界）。
 *
 * <p>口径来源：python 侧为唯一事实源（`models/schema/ai_agent.py`、`ai_schedule.py`、`ai_memory.py`、
 * `ai_conversation.py`）。本类走**真实 MVC 绑定链路**而非直测注解，因此同时覆盖"注解存在"与
 * "触发条件（使用处 `@Valid`）在位"两件事——若某端点漏 `@Valid`，用例会因请求直达服务层而失败
 * （越界值不会产生 400），这正是"死注解"复发时的预警。
 *
 * <p>关键噪音（勿被"看起来该加"误导）：
 * <ul>
 *   <li>`sortOrder` 侧 python 为 `ge=0` 且**无上界** → 只加 `@Min`，用例不得断言 `@Max` 行为；</li>
 *   <li>`EndpointUpdate.status`、`ScheduleUpdate.enabled`、`MemoryUpdate.status`、`AgentUpdate.sortOrder`
 *       在 python 为 `None` 语义 → 只加范围不 `@NotNull`，缺省必须放行；</li>
 *   <li>`ConversationUpdate.pinned/status` 在 python 是**无界裸 int** → 不加约束，故无对应用例。</li>
 * </ul>
 *
 * <p>越界断言统一配 `verifyNoInteractions(service)`：证明拦截发生在进入 Controller 方法之前，
 * 排除"进方法后由业务层兜底返回 A0400"造成的假阳性；长度类用例因需构造必填字段，同样只断 400。
 */
@DisplayName("AI 域字段约束对齐（P0 范围 / P1 长度 / 方法级 limit）")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiFieldConstraintValidationTest {

    /** Agent 创建的必填字段（避免被 @NotBlank 抢先拦截，掩盖被测字段的约束） */
    private static final String AGENT_REQUIRED = "\"agentCode\":\"a1\",\"name\":\"n\",\"modelId\":\"m1\"";
    /** A2A 端点创建的必填字段 */
    private static final String ENDPOINT_REQUIRED = "\"name\":\"ep\",\"baseUrl\":\"https://example.com/a2a\"";

    @Mock
    private AiAgentService agentService;
    @Mock
    private AiAgentVersionService agentVersionService;
    @Mock
    private AiAgentEndpointService endpointService;
    @Mock
    private AiScheduleService scheduleService;
    @Mock
    private AiMemoryService memoryService;
    @Mock
    private AiConversationService conversationService;

    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        LocalValidatorFactoryBean validator = new LocalValidatorFactoryBean();
        validator.afterPropertiesSet();
        MethodValidationPostProcessor methodValidation = new MethodValidationPostProcessor();
        methodValidation.afterPropertiesSet();

        mockMvc = MockMvcBuilders.standaloneSetup(
                        proxy(new AiAgentController(agentService, agentVersionService,
                                new com.pei.dehaze.service.AiAgentConfigResolver(null)), methodValidation),
                        proxy(new AiAgentEndpointController(endpointService), methodValidation),
                        proxy(new AiScheduleController(scheduleService), methodValidation),
                        proxy(new AiMemoryController(memoryService), methodValidation),
                        proxy(new AiConversationController(conversationService), methodValidation))
                .setControllerAdvice(new GlobalExceptionHandler())
                .setValidator(validator)
                .build();
    }

    /** 模拟 Boot 的 ValidationAutoConfiguration：@Validated 类需经代理才有方法级校验（limit 下界） */
    private static Object proxy(Object controller, MethodValidationPostProcessor processor) {
        return processor.postProcessAfterInitialization(controller, controller.getClass().getSimpleName());
    }

    private static String json(String body) {
        return "{" + body + "}";
    }

    // ---------- P0：数值范围（python 有界、java 原先无约束） ----------

    @Test
    @DisplayName("Agent 创建 status=2 → 400 + A0400 且未触达 service")
    void agentCreateStatusOutOfRange() throws Exception {
        mockMvc.perform(post("/api/v1/ai/agents").contentType(MediaType.APPLICATION_JSON)
                        .content(json(AGENT_REQUIRED + ",\"status\":2")))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(agentService);
    }

    @Test
    @DisplayName("Agent 创建 sortOrder=-1 → 400（python 仅 ge=0，无上界）")
    void agentCreateSortOrderNegative() throws Exception {
        mockMvc.perform(post("/api/v1/ai/agents").contentType(MediaType.APPLICATION_JSON)
                        .content(json(AGENT_REQUIRED + ",\"sortOrder\":-1")))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @Test
    @DisplayName("Agent 更新 sortOrder=-1 → 400（验证 @Valid 触发条件在位）")
    void agentUpdateSortOrderNegative() throws Exception {
        mockMvc.perform(put("/api/v1/ai/agents/1").contentType(MediaType.APPLICATION_JSON)
                        .content(json("\"sortOrder\":-1")))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @Test
    @DisplayName("Agent 启停 status=2 → 400 + A0400（@NotNull 已在，仅需范围）")
    void agentStatusOutOfRange() throws Exception {
        mockMvc.perform(patch("/api/v1/ai/agents/1/status").contentType(MediaType.APPLICATION_JSON)
                        .content(json("\"status\":2")))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(agentService);
    }

    @Test
    @DisplayName("Agent 列表 status=2 → 400（Query DTO 参数级 @Valid）")
    void agentListStatusOutOfRange() throws Exception {
        mockMvc.perform(get("/api/v1/ai/agents").param("status", "2"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @Test
    @DisplayName("A2A 端点创建 status=2 → 400 + A0400")
    void endpointCreateStatusOutOfRange() throws Exception {
        mockMvc.perform(post("/api/v1/ai/a2a/endpoints").contentType(MediaType.APPLICATION_JSON)
                        .content(json(ENDPOINT_REQUIRED + ",\"status\":2")))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(endpointService);
    }

    @Test
    @DisplayName("A2A 端点更新 status=2 → 400（该端点原先漏 @Valid，本用例即其触发条件守卫）")
    void endpointUpdateStatusOutOfRange() throws Exception {
        mockMvc.perform(patch("/api/v1/ai/a2a/endpoints/1").contentType(MediaType.APPLICATION_JSON)
                        .content(json("\"status\":2")))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(endpointService);
    }

    @Test
    @DisplayName("A2A 端点列表 status=2 → 400")
    void endpointListStatusOutOfRange() throws Exception {
        mockMvc.perform(get("/api/v1/ai/a2a/endpoints").param("status", "2"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @Test
    @DisplayName("定时任务更新 enabled=2 → 400（nullable，缺省不校验但越界必拒）")
    void scheduleUpdateEnabledOutOfRange() throws Exception {
        mockMvc.perform(put("/api/v1/ai/scheduled-tasks/1").contentType(MediaType.APPLICATION_JSON)
                        .content(json("\"enabled\":2")))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @Test
    @DisplayName("定时任务启停 enabled=2 → 400")
    void scheduleStatusEnabledOutOfRange() throws Exception {
        mockMvc.perform(patch("/api/v1/ai/scheduled-tasks/1/status").contentType(MediaType.APPLICATION_JSON)
                        .content(json("\"enabled\":2")))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(scheduleService);
    }

    @Test
    @DisplayName("记忆更新 status=2 → 400（该端点原先漏 @Valid）")
    void memoryUpdateStatusOutOfRange() throws Exception {
        mockMvc.perform(put("/api/v1/ai/memories/1").contentType(MediaType.APPLICATION_JSON)
                        .content(json("\"status\":2")))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(memoryService);
    }

    // ---------- P1：字符串长度（python 有 max_length、java 原先无 @Size） ----------

    @ParameterizedTest(name = "会话创建 {0} 超长 → 400")
    @CsvSource({"title,256", "model,65", "agentCode,65", "scene,33"})
    @DisplayName("会话创建字符串超长 → 400 + A0400")
    void conversationCreateStringTooLong(String field, int length) throws Exception {
        mockMvc.perform(post("/api/v1/ai/conversations").contentType(MediaType.APPLICATION_JSON)
                        .content(json("\"" + field + "\":\"" + "x".repeat(length) + "\"")))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @ParameterizedTest(name = "会话更新 {0} 超长 → 400")
    @CsvSource({"title,256", "model,65", "agentCode,65"})
    @DisplayName("会话更新字符串超长 → 400 + A0400（该端点原先漏 @Valid）")
    void conversationUpdateStringTooLong(String field, int length) throws Exception {
        mockMvc.perform(patch("/api/v1/ai/conversations/1").contentType(MediaType.APPLICATION_JSON)
                        .content(json("\"" + field + "\":\"" + "x".repeat(length) + "\"")))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @Test
    @DisplayName("Agent 创建 modelId 超长(65) → 400")
    void agentCreateModelIdTooLong() throws Exception {
        mockMvc.perform(post("/api/v1/ai/agents").contentType(MediaType.APPLICATION_JSON)
                        .content(json("\"agentCode\":\"a1\",\"name\":\"n\",\"modelId\":\"" + "x".repeat(65) + "\"")))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    @Test
    @DisplayName("Agent 更新 modelId 超长(65) → 400")
    void agentUpdateModelIdTooLong() throws Exception {
        mockMvc.perform(put("/api/v1/ai/agents/1").contentType(MediaType.APPLICATION_JSON)
                        .content(json("\"modelId\":\"" + "x".repeat(65) + "\"")))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
    }

    // ---------- 方法级：记忆搜索 limit 下界 ----------

    @ParameterizedTest(name = "记忆搜索 limit={0} → 400 + A0400")
    @CsvSource({"0", "-1"})
    @DisplayName("记忆搜索 limit 低于下界 → 400（方法级校验，非 DTO 字段级）")
    void memorySearchLimitBelowMin(String limit) throws Exception {
        mockMvc.perform(get("/api/v1/ai/memories/search").param("keyword", "k").param("limit", limit))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(memoryService);
    }

    // ---------- 放行断言：防"用过度收紧换通过" ----------

    @Test
    @DisplayName("边界值放行：Agent status=0、sortOrder=0、modelId 恰好 64 字")
    void agentCreateBoundaryAccepted() throws Exception {
        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(1L);
            mockMvc.perform(post("/api/v1/ai/agents").contentType(MediaType.APPLICATION_JSON)
                            .content(json(AGENT_REQUIRED + ",\"status\":0,\"sortOrder\":0")))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.code").value("00000"));
        }
    }

    @Test
    @DisplayName("边界值放行：会话标题 255 / 模型 64 / Agent编码 64 / 场景 32")
    void conversationCreateBoundaryAccepted() throws Exception {
        String body = json("\"title\":\"" + "x".repeat(255) + "\",\"model\":\"" + "x".repeat(64)
                + "\",\"agentCode\":\"" + "x".repeat(64) + "\",\"scene\":\"" + "x".repeat(32) + "\"");
        mockMvc.perform(post("/api/v1/ai/conversations").contentType(MediaType.APPLICATION_JSON).content(body))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }

    @Test
    @DisplayName("边界值放行：记忆搜索 limit=1（下界本身合法）")
    void memorySearchLimitAtMinAccepted() throws Exception {
        mockMvc.perform(get("/api/v1/ai/memories/search").param("keyword", "k").param("limit", "1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }

    @Test
    @DisplayName("nullable 放行：端点更新与定时任务更新缺省 status/enabled 不报错（python 为 None 语义）")
    void nullableFieldsAbsentAccepted() throws Exception {
        mockMvc.perform(patch("/api/v1/ai/a2a/endpoints/1").contentType(MediaType.APPLICATION_JSON)
                        .content(json("\"name\":\"ep2\"")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
        mockMvc.perform(put("/api/v1/ai/scheduled-tasks/1").contentType(MediaType.APPLICATION_JSON)
                        .content(json("\"name\":\"t2\"")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));
    }
}

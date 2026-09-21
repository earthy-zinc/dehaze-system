package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.baomidou.mybatisplus.core.toolkit.GlobalConfigUtils;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.AiObservabilityMapper;
import com.pei.dehaze.mapper.SysAiAgentThoughtMapper;
import com.pei.dehaze.mapper.SysAiArtifactMapper;
import com.pei.dehaze.mapper.SysAiBillingMapper;
import com.pei.dehaze.mapper.SysAiConversationMapper;
import com.pei.dehaze.mapper.SysAiLlmCallMapper;
import com.pei.dehaze.mapper.SysAiMessageMapper;
import com.pei.dehaze.mapper.SysAiTraceMapper;
import com.pei.dehaze.model.entity.SysAiAgentThought;
import com.pei.dehaze.model.entity.SysAiBilling;
import com.pei.dehaze.model.entity.SysAiConversation;
import com.pei.dehaze.model.entity.SysAiLlmCall;
import com.pei.dehaze.model.entity.SysAiMessage;
import com.pei.dehaze.model.entity.SysAiTrace;
import com.pei.dehaze.model.query.AiObservabilityCostsQuery;
import com.pei.dehaze.model.query.AiObservabilityTraceQuery;
import com.pei.dehaze.model.query.AiObservabilityTrendsQuery;
import com.pei.dehaze.model.read.AiTraceCostRead;
import com.pei.dehaze.model.read.AiTraceTrendRead;
import com.pei.dehaze.model.vo.AiObservabilityTimelineVO;
import com.pei.dehaze.model.vo.AiObservabilityTraceDetailVO;
import com.pei.dehaze.model.vo.AiObservabilityTraceItemVO;
import com.pei.dehaze.model.vo.AiObservabilityTrendVO;
import com.pei.dehaze.security.util.SecurityUtils;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * AiObservabilityService 单元测试。
 *
 * <p>锁定与 dehaze-python 一致的关键口径：总览计数、过程链归属校验（A0401 不暴露存在性）、
 * 时间线事件交织与 raw 开关、消耗/趋势聚合、导出限行与 UTF-8 BOM。
 */
@DisplayName("AiObservabilityService 单元测试")
@ExtendWith(MockitoExtension.class)
class AiObservabilityServiceTest {

    @Mock
    private AiObservabilityMapper observabilityMapper;

    @Mock
    private SysAiTraceMapper traceMapper;

    @Mock
    private SysAiLlmCallMapper llmCallMapper;

    @Mock
    private SysAiAgentThoughtMapper thoughtMapper;

    @Mock
    private SysAiArtifactMapper artifactMapper;

    @Mock
    private SysAiMessageMapper messageMapper;

    @Mock
    private SysAiConversationMapper conversationMapper;

    @Mock
    private SysAiBillingMapper billingMapper;

    private AiObservabilityService service;

    /** 无 Spring 上下文时 LambdaQueryWrapper 的 select(...) 需要先注册实体表信息 */
    @BeforeAll
    static void initTableInfo() {
        MybatisConfiguration configuration = new MybatisConfiguration();
        GlobalConfigUtils.setGlobalConfig(configuration, GlobalConfigUtils.defaults());
        TableInfoHelper.initTableInfo(new MapperBuilderAssistant(configuration, ""), SysAiConversation.class);
    }

    @BeforeEach
    void setUp() {
        service = new AiObservabilityService(observabilityMapper, traceMapper, llmCallMapper, thoughtMapper,
                artifactMapper, messageMapper, conversationMapper, billingMapper, new ObjectMapper());
    }

    @Test
    @DisplayName("异常总览：状态分布 + 配额拒绝 + 高风险调用计数")
    void summary_aggregatesCounts() {
        when(observabilityMapper.countByStatus(1)).thenReturn(10L);
        when(observabilityMapper.countByStatus(2)).thenReturn(2L);
        when(observabilityMapper.countByStatus(3)).thenReturn(1L);
        when(observabilityMapper.countByStatus(4)).thenReturn(3L);
        when(observabilityMapper.countQuotaRejected(any())).thenReturn(4L);
        when(observabilityMapper.countHighRisk(40)).thenReturn(1L);

        var vo = service.summary();

        assertThat(vo.getTotal()).isEqualTo(16L);
        assertThat(vo.getSuccessCount()).isEqualTo(10L);
        assertThat(vo.getTimeoutCount()).isEqualTo(3L);
        assertThat(vo.getQuotaRejected()).isEqualTo(4L);
        assertThat(vo.getHighRiskCalls()).isEqualTo(1L);
    }

    @Test
    @DisplayName("过程链检索：非法 capability 抛 A0400")
    void listTraces_rejectsUnknownCapability() {
        AiObservabilityTraceQuery query = new AiObservabilityTraceQuery();
        query.setCapability("voice");

        assertThatThrownBy(() -> service.listTraces(query))
                .isInstanceOf(BusinessException.class)
                .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                .isEqualTo(ResultCode.PARAM_ERROR.getCode());
        verify(observabilityMapper, never()).countTraces(any(), any(), any());
    }

    @Test
    @DisplayName("过程链检索：回填会话标题并透出 TRACE 汇总字段")
    void listTraces_fillsConversationTitle() {
        SysAiTrace trace = trace("trace-1", 5L, 7L);
        when(observabilityMapper.countTraces(any(), any(), any())).thenReturn(1L);
        when(observabilityMapper.selectTracePage(any(), any(), any(), eq(0L), eq(10L)))
                .thenReturn(List.of(trace));
        SysAiConversation conversation = new SysAiConversation();
        conversation.setId(5L);
        conversation.setTitle("灰度测试会话");
        when(conversationMapper.selectList(any())).thenReturn(List.of(conversation));

        IPage<AiObservabilityTraceItemVO> page = service.listTraces(new AiObservabilityTraceQuery());

        assertThat(page.getTotal()).isEqualTo(1);
        assertThat(page.getRecords().get(0).getConversationTitle()).isEqualTo("灰度测试会话");
        assertThat(page.getRecords().get(0).getTraceType()).isEqualTo("conversation");
    }

    @Test
    @DisplayName("过程链详情：不存在抛 A0401")
    void getTrace_notFound() {
        when(traceMapper.selectOne(any())).thenReturn(null);

        assertThatThrownBy(() -> service.getTrace("missing"))
                .isInstanceOf(BusinessException.class)
                .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                .isEqualTo(ResultCode.RESOURCE_NOT_FOUND.getCode());
    }

    @Test
    @DisplayName("过程链详情：普通用户查他人会话一律 A0401（不暴露存在性）")
    void getTrace_hidesForeignConversationFromNormalUser() {
        when(traceMapper.selectOne(any())).thenReturn(trace("trace-1", 5L, 7L));
        SysAiConversation conversation = new SysAiConversation();
        conversation.setId(5L);
        conversation.setUserId(99L);
        when(conversationMapper.selectById(5L)).thenReturn(conversation);
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::isRoot).thenReturn(false);
            security.when(SecurityUtils::getPerms).thenReturn(Set.of());
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            assertThatThrownBy(() -> service.getTrace("trace-1"))
                    .isInstanceOf(BusinessException.class)
                    .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                    .isEqualTo(ResultCode.RESOURCE_NOT_FOUND.getCode());
        }
        verify(llmCallMapper, never()).selectList(any());
    }

    @Test
    @DisplayName("过程链详情：审计权限全量可见，计费按 request_id 无命中回退 message_id")
    void getTrace_adminAssemblesDetailWithBillingFallback() {
        SysAiTrace trace = trace("trace-1", 5L, 7L);
        trace.setContextSnapshot("{\"items\":[{\"type\":\"memory\",\"tokens\":10}]}");
        trace.setErrorDetail("{\"message\":\"中断\"}");
        when(traceMapper.selectOne(any())).thenReturn(trace);
        SysAiLlmCall call = new SysAiLlmCall();
        call.setTraceId("trace-1");
        call.setSeq(1);
        call.setModel("gpt-4o");
        call.setStatus(1);
        call.setToolCall("{\"has_tool_call\":true}");
        when(llmCallMapper.selectList(any())).thenReturn(List.of(call));
        SysAiAgentThought thought = new SysAiAgentThought();
        thought.setId(3L);
        thought.setMessageId(7L);
        thought.setPosition(1);
        thought.setTool("predict");
        when(thoughtMapper.selectList(any())).thenReturn(List.of(thought));
        SysAiBilling billing = new SysAiBilling();
        billing.setId(11L);
        billing.setMessageId(7L);
        billing.setCredits(20);
        when(billingMapper.selectList(any())).thenReturn(List.of(), List.of(billing));
        SysAiMessage message = new SysAiMessage();
        message.setId(7L);
        message.setConversationId(5L);
        message.setRole("assistant");
        when(messageMapper.selectList(any())).thenReturn(List.of(message));

        AiObservabilityTraceDetailVO detail;
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::isRoot).thenReturn(false);
            security.when(SecurityUtils::getPerms).thenReturn(Set.of("ai:conversation:audit"));
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            detail = service.getTrace("trace-1");
        }

        assertThat(detail.getLlmCalls()).hasSize(1);
        assertThat(detail.getLlmCalls().get(0).getToolCall())
                .isEqualTo(Map.of("has_tool_call", true));
        assertThat(detail.getThoughts()).hasSize(1);
        assertThat(detail.getThoughts().get(0).getTool()).isEqualTo("predict");
        assertThat(detail.getBilling()).hasSize(1);
        assertThat(detail.getBilling().get(0).getCredits()).isEqualTo(20);
        assertThat(detail.getMessages()).hasSize(1);
        assertThat(detail.getContextSnapshot()).isInstanceOf(Map.class);
        assertThat(detail.getErrorDetail()).isEqualTo(Map.of("message", "中断"));
    }

    @Test
    @DisplayName("时间线：普通用户查非本人会话抛 A0401")
    void getTimeline_hidesForeignConversation() {
        when(conversationMapper.selectOne(any())).thenReturn(null);
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::isRoot).thenReturn(false);
            security.when(SecurityUtils::getPerms).thenReturn(Set.of());
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            assertThatThrownBy(() -> service.getTimeline(5L, null))
                    .isInstanceOf(BusinessException.class)
                    .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                    .isEqualTo(ResultCode.RESOURCE_NOT_FOUND.getCode());
        }
    }

    @Test
    @DisplayName("时间线：轮次切分 + 事件按 ts/业务序交织 + include 控制 raw 报文")
    void getTimeline_buildsRoundsAndEvents() {
        SysAiConversation conversation = new SysAiConversation();
        conversation.setId(5L);
        conversation.setUserId(1L);
        conversation.setTitle("会话");
        conversation.setCurrentBranchMessageId(2L);
        when(conversationMapper.selectById(5L)).thenReturn(conversation);

        LocalDateTime base = LocalDateTime.of(2026, 9, 17, 10, 0, 0);
        SysAiMessage user = message(1L, "user", base, null);
        SysAiMessage assistant = message(2L, "assistant", base.plusSeconds(8), 1L);
        when(messageMapper.selectList(any())).thenReturn(List.of(user, assistant));

        SysAiTrace trace = trace("trace-1", 5L, 2L);
        trace.setDurationMs(8000);
        trace.setCreateTime(base.plusSeconds(8));
        trace.setContextSnapshot("{\"events\":[{\"event\":\"summarize\",\"tokens\":100}]}");
        when(traceMapper.selectList(any())).thenReturn(List.of(trace));

        SysAiLlmCall call = new SysAiLlmCall();
        call.setTraceId("trace-1");
        call.setSeq(1);
        call.setStatus(1);
        call.setStartTime(base.plusSeconds(1));
        call.setRawRequest("{\"model\":\"gpt-4o\"}");
        when(llmCallMapper.selectList(any())).thenReturn(List.of(call));

        SysAiAgentThought thought = new SysAiAgentThought();
        thought.setId(3L);
        thought.setMessageId(2L);
        thought.setPosition(1);
        thought.setTool("predict");
        thought.setCreateTime(base.plusSeconds(3));
        when(thoughtMapper.selectList(any())).thenReturn(List.of(thought));

        SysAiBilling billing = new SysAiBilling();
        billing.setId(11L);
        billing.setRequestId("trace-1");
        billing.setBillType("chat");
        billing.setCredits(20);
        billing.setInputTokens(100);
        billing.setOutputTokens(50);
        billing.setCachedInputTokens(10);
        billing.setCreateTime(base.plusSeconds(4));
        when(billingMapper.selectList(any())).thenReturn(List.of(billing));

        AiObservabilityTimelineVO timeline;
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::isRoot).thenReturn(true);
            security.when(SecurityUtils::getPerms).thenReturn(Set.of());
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            timeline = service.getTimeline(5L, null);
        }

        assertThat(timeline.getConversation().getId()).isEqualTo(5L);
        assertThat(timeline.getRounds()).hasSize(1);
        List<AiObservabilityTimelineVO.Event> events = timeline.getRounds().get(0).getTraces().get(0).getEvents();
        // 同刻事件按业务序：input(0) → context(1) → system_event(2) → llm_call(3) → tool_exec(4) → billing(5)
        assertThat(events).extracting(AiObservabilityTimelineVO.Event::getKind)
                .containsExactly("input", "context", "system_event", "llm_call", "tool_exec", "billing");
        AiObservabilityTimelineVO.Event inputEvent = events.get(0);
        assertThat(inputEvent.getMessage().getId()).isEqualTo(1L);
        AiObservabilityTimelineVO.Event llmEvent = events.get(3);
        assertThat(llmEvent.getRawRequest()).isEqualTo(Map.of("model", "gpt-4o"));
        @SuppressWarnings("unchecked")
        Map<String, Object> summary = (Map<String, Object>) llmEvent.getSummary();
        assertThat(summary).containsEntry("outputSnapshot", null);
        AiObservabilityTimelineVO.Event billingEvent = events.get(5);
        assertThat(billingEvent.getTokens()).isEqualTo(Map.of("input", 100, "output", 50, "cached", 10));
    }

    @Test
    @DisplayName("时间线：include 非 raw 时省略 wire 原始报文，summary 仅保留计数形状")
    void getTimeline_omitsRawWhenNotRequested() {
        SysAiConversation conversation = new SysAiConversation();
        conversation.setId(5L);
        conversation.setUserId(1L);
        conversation.setCurrentBranchMessageId(2L);
        when(conversationMapper.selectById(5L)).thenReturn(conversation);
        LocalDateTime base = LocalDateTime.of(2026, 9, 17, 10, 0, 0);
        when(messageMapper.selectList(any()))
                .thenReturn(List.of(message(1L, "user", base, null), message(2L, "assistant", base, 1L)));
        SysAiTrace trace = trace("trace-1", 5L, 2L);
        trace.setDurationMs(1000);
        trace.setCreateTime(base);
        when(traceMapper.selectList(any())).thenReturn(List.of(trace));
        SysAiLlmCall call = new SysAiLlmCall();
        call.setTraceId("trace-1");
        call.setSeq(1);
        call.setStatus(1);
        call.setStartTime(base);
        call.setRawRequest("{\"model\":\"gpt-4o\"}");
        call.setRawResponse("{\"id\":\"1\"}");
        call.setInputSnapshot("{\"messages\":{\"counts\":{\"user\":1},\"tokens\":10,\"items\":["
                + "{\"role\":\"user\",\"content\":\"全文\"}]},\"system_tokens\":5,\"tools\":[{\"name\":\"x\"}]}");
        when(llmCallMapper.selectList(any())).thenReturn(List.of(call));
        when(thoughtMapper.selectList(any())).thenReturn(List.of());
        when(billingMapper.selectList(any())).thenReturn(List.of());

        AiObservabilityTimelineVO timeline;
        try (MockedStatic<SecurityUtils> security = mockStatic(SecurityUtils.class)) {
            security.when(SecurityUtils::isRoot).thenReturn(true);
            security.when(SecurityUtils::getPerms).thenReturn(Set.of());
            security.when(SecurityUtils::getUserId).thenReturn(1L);

            timeline = service.getTimeline(5L, "summary");
        }

        List<AiObservabilityTimelineVO.Event> events = timeline.getRounds().get(0).getTraces().get(0).getEvents();
        AiObservabilityTimelineVO.Event llmEvent = events.stream()
                .filter(event -> "llm_call".equals(event.getKind())).findFirst().orElseThrow();
        assertThat(llmEvent.getRawRequest()).isNull();
        assertThat(llmEvent.getRawResponse()).isNull();
        @SuppressWarnings("unchecked")
        Map<String, Object> summary = (Map<String, Object>) llmEvent.getSummary();
        assertThat(summary).containsKeys("inputSnapshot", "outputSnapshot");
        @SuppressWarnings("unchecked")
        Map<String, Object> slim = (Map<String, Object>) summary.get("inputSnapshot");
        assertThat(slim).containsEntry("system_tokens", 5).doesNotContainKey("tools");
        @SuppressWarnings("unchecked")
        Map<String, Object> messages = (Map<String, Object>) slim.get("messages");
        assertThat(messages).containsKeys("counts", "tokens").doesNotContainKey("items");
    }

    @Test
    @DisplayName("资源消耗聚合：按维度映射字段并透出按日趋势")
    void costs_mapsDimensionFields() {
        AiTraceCostRead row = new AiTraceCostRead();
        row.setDimension("gpt-4o");
        row.setTraceCount(3L);
        row.setTotalTokens(300L);
        row.setPromptTokens(200L);
        row.setCompletionTokens(100L);
        row.setCachedTokens(20L);
        when(observabilityMapper.countCostGroups(eq("model"), any(), any())).thenReturn(1L);
        when(observabilityMapper.selectCostRows(eq("model"), any(), any(), anyLong(), anyLong()))
                .thenReturn(List.of(row));
        when(observabilityMapper.selectCostTrend(eq("model"), any(), any())).thenReturn(List.of());

        AiObservabilityCostsQuery query = new AiObservabilityCostsQuery();
        var result = service.costs(query);

        assertThat(result.getTotal()).isEqualTo(1L);
        assertThat(result.getItems().get(0).getModel()).isEqualTo("gpt-4o");
        assertThat(result.getItems().get(0).getTotalTokens()).isEqualTo(300L);
        assertThat(result.getItems().get(0).getUserId()).isNull();
    }

    @Test
    @DisplayName("资源消耗聚合：user 维度透出用户ID，非法维度抛 A0400")
    void costs_validatesDimension() {
        AiObservabilityCostsQuery invalid = new AiObservabilityCostsQuery();
        invalid.setDimension("provider");
        assertThatThrownBy(() -> service.costs(invalid))
                .isInstanceOf(BusinessException.class)
                .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                .isEqualTo(ResultCode.PARAM_ERROR.getCode());

        AiTraceCostRead row = new AiTraceCostRead();
        row.setDimension("9");
        row.setTraceCount(1L);
        row.setTotalTokens(10L);
        when(observabilityMapper.countCostGroups(eq("user"), any(), any())).thenReturn(1L);
        when(observabilityMapper.selectCostRows(eq("user"), any(), any(), anyLong(), anyLong()))
                .thenReturn(List.of(row));
        when(observabilityMapper.selectCostTrend(eq("user"), any(), any())).thenReturn(List.of());
        AiObservabilityCostsQuery query = new AiObservabilityCostsQuery();
        query.setDimension("user");

        assertThat(service.costs(query).getItems().get(0).getUserId()).isEqualTo(9L);
    }

    @Test
    @DisplayName("性能趋势：成功率百分比与平均延迟保留两位，首Token取成功调用口径")
    void trends_computesRates() {
        AiTraceTrendRead row = new AiTraceTrendRead();
        row.setDimension("gpt-4o");
        row.setDate("2026-09-17");
        row.setCallCount(4L);
        row.setSuccessCount(3L);
        row.setAvgFirstTokenMs(new BigDecimal("123.456"));
        row.setAvgDurationMs(new BigDecimal("456.789"));
        when(observabilityMapper.selectTrends(eq("model"), any(), any())).thenReturn(List.of(row));

        List<AiObservabilityTrendVO> items = service.trends(new AiObservabilityTrendsQuery());

        assertThat(items).hasSize(1);
        assertThat(items.get(0).getSuccessRate()).isEqualTo(75.0);
        assertThat(items.get(0).getAvgFirstTokenMs()).isEqualTo(123.46);
        assertThat(items.get(0).getAvgDurationMs()).isEqualTo(456.79);
        assertThat(items.get(0).getAgentCode()).isNull();
    }

    @Test
    @DisplayName("性能趋势：非法维度抛 A0400")
    void trends_rejectsUnknownDimension() {
        AiObservabilityTrendsQuery query = new AiObservabilityTrendsQuery();
        query.setDimension("user");

        assertThatThrownBy(() -> service.trends(query))
                .isInstanceOf(BusinessException.class)
                .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                .isEqualTo(ResultCode.PARAM_ERROR.getCode());
    }

    @Test
    @DisplayName("过程链导出：超出行数限制抛 A0709")
    void exportTraces_rejectsOverLimit() {
        when(observabilityMapper.countTraces(any(), any(), any())).thenReturn(100_001L);

        assertThatThrownBy(() -> service.exportTraces(new AiObservabilityTraceQuery()))
                .isInstanceOf(BusinessException.class)
                .extracting(e -> ((BusinessException) e).getResultCode().getCode())
                .isEqualTo(ResultCode.EXPORT_ROWS_EXCEED_LIMIT.getCode());
        verify(observabilityMapper, never()).selectTraces(any(), any(), any());
    }

    @Test
    @DisplayName("过程链导出：CSV 带 UTF-8 BOM 与固定表头")
    void exportTraces_prefixesBom() {
        when(observabilityMapper.countTraces(any(), any(), any())).thenReturn(1L);
        SysAiTrace trace = trace("trace-1", 5L, 7L);
        trace.setCreateTime(LocalDateTime.of(2026, 9, 17, 10, 0, 0));
        when(observabilityMapper.selectTraces(any(), any(), any())).thenReturn(List.of(trace));

        byte[] csv = service.exportTraces(new AiObservabilityTraceQuery());

        assertThat(csv[0] & 0xFF).isEqualTo(0xEF);
        assertThat(csv[1] & 0xFF).isEqualTo(0xBB);
        assertThat(csv[2] & 0xFF).isEqualTo(0xBF);
        String text = new String(csv, 3, csv.length - 3, java.nio.charset.StandardCharsets.UTF_8);
        assertThat(text).startsWith("trace_id,conversation_id,message_id,agent_code,model,status,error_type");
        assertThat(text).contains("trace-1,5,7,agent-a,gpt-4o,1,,8000,120,1,300,200,100,50,2,2026-09-17 10:00:00");
    }

    private static SysAiTrace trace(String traceId, Long conversationId, Long messageId) {
        SysAiTrace trace = new SysAiTrace();
        trace.setId(1L);
        trace.setTraceId(traceId);
        trace.setConversationId(conversationId);
        trace.setMessageId(messageId);
        trace.setAgentCode("agent-a");
        trace.setTraceType("conversation");
        trace.setModel("gpt-4o");
        trace.setStatus(1);
        trace.setDurationMs(8000);
        trace.setFirstTokenMs(120);
        trace.setLlmCallCount(1);
        trace.setTotalTokens(300);
        trace.setPromptTokens(200);
        trace.setCompletionTokens(100);
        trace.setCachedTokens(50);
        trace.setStepCount(2);
        trace.setCreateTime(LocalDateTime.of(2026, 9, 17, 10, 0, 0));
        return trace;
    }

    private static SysAiMessage message(Long id, String role, LocalDateTime createTime, Long parentId) {
        SysAiMessage message = new SysAiMessage();
        message.setId(id);
        message.setConversationId(5L);
        message.setRole(role);
        message.setContent(role + "-content");
        message.setStatus(2);
        message.setParentMessageId(parentId);
        message.setCreateTime(createTime);
        return message;
    }
}

package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.AiDateTimeUtils;
import com.pei.dehaze.mapper.AiObservabilityMapper;
import com.pei.dehaze.mapper.SysAiAgentThoughtMapper;
import com.pei.dehaze.mapper.SysAiArtifactMapper;
import com.pei.dehaze.mapper.SysAiBillingMapper;
import com.pei.dehaze.mapper.SysAiConversationMapper;
import com.pei.dehaze.mapper.SysAiLlmCallMapper;
import com.pei.dehaze.mapper.SysAiMessageMapper;
import com.pei.dehaze.mapper.SysAiTraceMapper;
import com.pei.dehaze.model.entity.SysAiAgentThought;
import com.pei.dehaze.model.entity.SysAiArtifact;
import com.pei.dehaze.model.entity.SysAiBilling;
import com.pei.dehaze.model.entity.SysAiConversation;
import com.pei.dehaze.model.entity.SysAiLlmCall;
import com.pei.dehaze.model.entity.SysAiMessage;
import com.pei.dehaze.model.entity.SysAiTrace;
import com.pei.dehaze.model.query.AiObservabilityCostsQuery;
import com.pei.dehaze.model.query.AiObservabilityTraceQuery;
import com.pei.dehaze.model.query.AiObservabilityTrendsQuery;
import com.pei.dehaze.model.read.AiTraceCostRead;
import com.pei.dehaze.model.read.AiTraceCostTrendRead;
import com.pei.dehaze.model.read.AiTraceTrendRead;
import com.pei.dehaze.model.vo.AiAgentThoughtVO;
import com.pei.dehaze.model.vo.AiObservabilityCostsVO;
import com.pei.dehaze.model.vo.AiObservabilitySummaryVO;
import com.pei.dehaze.model.vo.AiObservabilityTimelineVO;
import com.pei.dehaze.model.vo.AiObservabilityTraceDetailVO;
import com.pei.dehaze.model.vo.AiObservabilityTraceItemVO;
import com.pei.dehaze.model.vo.AiObservabilityTrendVO;
import com.pei.dehaze.security.util.SecurityUtils;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * AI 可观测性查询服务（F-M08-013）：异常总览、过程链检索/详情/导出、会话审计时间线、
 * 资源消耗聚合与性能趋势。
 *
 * <p>明细与聚合均以 dehaze-python {@code ai_observability_service} 为事实源：过程链详情管理员
 * 全量可见、普通用户仅可查自己会话（越权与不存在一律 A0401，不暴露存在性）。
 *
 * @author dehaze
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AiObservabilityService {

    private static final String AUDIT_PERMISSION = "ai:conversation:audit";

    /** 采集链路写入的配额拒绝类 error_type（中断类型 quota / 计费拒绝收尾路径） */
    private static final List<String> QUOTA_REJECT_ERROR_TYPES = List.of(
            "quota", "quota_exceeded", "precharge_blocked", "arrears", "balance_exceeded");

    /** 高风险调用：推理步数超阈值（防循环观测） */
    private static final int HIGH_RISK_STEP_THRESHOLD = 40;

    private static final Set<String> CAPABILITIES = Set.of("memory", "kb", "tools");

    /** 轮内事件同类同刻的业务优先级（input→context→system_event→llm_call→tool_exec→billing） */
    private static final Map<String, Integer> EVENT_PRIORITY = Map.of(
            "input", 0, "context", 1, "system_event", 2, "llm_call", 3, "tool_exec", 4, "billing", 5);

    private static final DateTimeFormatter CSV_TIME = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private static final int TRACE_MESSAGE_LIMIT = 1000;

    private final AiObservabilityMapper observabilityMapper;

    private final SysAiTraceMapper traceMapper;

    private final SysAiLlmCallMapper llmCallMapper;

    private final SysAiAgentThoughtMapper thoughtMapper;

    private final SysAiArtifactMapper artifactMapper;

    private final SysAiMessageMapper messageMapper;

    private final SysAiConversationMapper conversationMapper;

    private final SysAiBillingMapper billingMapper;

    private final ObjectMapper objectMapper;

    public AiObservabilitySummaryVO summary() {
        long success = observabilityMapper.countByStatus(1);
        long failed = observabilityMapper.countByStatus(2);
        long interrupted = observabilityMapper.countByStatus(3);
        long timeout = observabilityMapper.countByStatus(4);
        AiObservabilitySummaryVO vo = new AiObservabilitySummaryVO();
        vo.setTotal(success + failed + interrupted + timeout);
        vo.setSuccessCount(success);
        vo.setFailedCount(failed);
        vo.setInterruptedCount(interrupted);
        vo.setTimeoutCount(timeout);
        vo.setQuotaRejected(observabilityMapper.countQuotaRejected(QUOTA_REJECT_ERROR_TYPES));
        vo.setHighRiskCalls(observabilityMapper.countHighRisk(HIGH_RISK_STEP_THRESHOLD));
        return vo;
    }

    public IPage<AiObservabilityTraceItemVO> listTraces(AiObservabilityTraceQuery query) {
        if (query.getCapability() != null && !CAPABILITIES.contains(query.getCapability())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "capability 仅支持 memory/kb/tools");
        }
        LocalDateTime start = AiDateTimeUtils.parse(query.getStartTime());
        LocalDateTime end = AiDateTimeUtils.parse(query.getEndTime());
        long total = observabilityMapper.countTraces(query, start, end);
        long offset = (long) (query.getPageNum() - 1) * query.getPageSize();
        List<SysAiTrace> traces = observabilityMapper.selectTracePage(query, start, end, offset, query.getPageSize());
        Map<Long, String> titles = conversationTitles(traces);
        List<AiObservabilityTraceItemVO> records = new ArrayList<>();
        for (SysAiTrace trace : traces) {
            AiObservabilityTraceItemVO vo = toItemVO(trace);
            vo.setConversationTitle(titles.get(trace.getConversationId()));
            records.add(vo);
        }
        return pageOf(query.getPageNum(), query.getPageSize(), records, total);
    }

    /**
     * 过程链详情：管理员全量可见；普通用户仅可查自己会话，跨会话访问与不存在一律 A0401
     */
    public AiObservabilityTraceDetailVO getTrace(String traceId) {
        SysAiTrace trace = traceMapper.selectOne(new LambdaQueryWrapper<SysAiTrace>()
                .eq(SysAiTrace::getTraceId, traceId).last("LIMIT 1"));
        if (trace == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "过程链不存在");
        }
        if (!isAudit() && !Objects.equals(conversationOwner(trace.getConversationId()), SecurityUtils.getUserId())) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "过程链不存在");
        }

        AiObservabilityTraceDetailVO detail = new AiObservabilityTraceDetailVO();
        copyTraceFields(trace, detail);
        detail.setContextSnapshot(readJson(trace.getContextSnapshot()));
        detail.setErrorDetail(readJson(trace.getErrorDetail()));
        detail.setLlmCalls(llmCallsByTrace(traceId));
        if (trace.getMessageId() != null) {
            detail.setThoughts(thoughtsByMessage(trace.getMessageId()));
            // 计费优先按 request_id=trace_id 精确归因，无命中回退 message_id
            List<SysAiBilling> billing = billingByRequestId(traceId);
            if (billing.isEmpty()) {
                billing = billingByMessage(trace.getMessageId());
            }
            detail.setBilling(billing.stream().map(this::toBillingVO).toList());
            detail.setArtifacts(artifactsByMessage(trace.getMessageId()));
        }
        detail.setMessages(messagesByConversation(trace.getConversationId()));
        return detail;
    }

    public AiObservabilityTimelineVO getTimeline(Long conversationId, String include) {
        return buildTimeline(conversationId, isAudit(), include == null
                || Arrays.stream(include.split(",")).anyMatch("raw"::equals));
    }

    /** 会话时间线导出：审计口径全量（含 raw 原始报文），camelCase JSON 整体下载 */
    public byte[] exportTimelineJson(Long conversationId) {
        AiObservabilityTimelineVO timeline = buildTimeline(conversationId, true, true);
        try {
            return objectMapper.writeValueAsBytes(timeline);
        } catch (Exception e) {
            throw new BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "时间线导出失败");
        }
    }

    public AiObservabilityCostsVO costs(AiObservabilityCostsQuery query) {
        String dimension = query.getDimension();
        if (!List.of("model", "agent", "user").contains(dimension)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "dimension 仅支持 model/agent/user");
        }
        LocalDateTime start = AiDateTimeUtils.parse(query.getStartTime());
        LocalDateTime end = AiDateTimeUtils.parse(query.getEndTime());
        long total = observabilityMapper.countCostGroups(dimension, start, end);
        long offset = (long) (query.getPageNum() - 1) * query.getPageSize();
        List<AiObservabilityCostsVO.Item> items = new ArrayList<>();
        for (AiTraceCostRead row : observabilityMapper.selectCostRows(dimension, start, end, offset, query.getPageSize())) {
            AiObservabilityCostsVO.Item item = new AiObservabilityCostsVO.Item();
            if ("model".equals(dimension)) {
                item.setModel(row.getDimension());
            } else if ("agent".equals(dimension)) {
                item.setAgentCode(row.getDimension());
            } else {
                item.setUserId(parseLong(row.getDimension()));
            }
            item.setTraceCount(row.getTraceCount());
            item.setTotalTokens(row.getTotalTokens());
            item.setPromptTokens(row.getPromptTokens());
            item.setCompletionTokens(row.getCompletionTokens());
            item.setCachedTokens(row.getCachedTokens());
            items.add(item);
        }
        List<AiObservabilityCostsVO.TrendItem> trend = new ArrayList<>();
        for (AiTraceCostTrendRead row : observabilityMapper.selectCostTrend(dimension, start, end)) {
            AiObservabilityCostsVO.TrendItem item = new AiObservabilityCostsVO.TrendItem();
            item.setDate(row.getDate());
            item.setTraceCount(row.getTraceCount());
            item.setTotalTokens(row.getTotalTokens());
            item.setPromptTokens(row.getPromptTokens());
            item.setCompletionTokens(row.getCompletionTokens());
            item.setCachedTokens(row.getCachedTokens());
            trend.add(item);
        }
        AiObservabilityCostsVO vo = new AiObservabilityCostsVO();
        vo.setItems(items);
        vo.setTotal(total);
        vo.setTrend(trend);
        return vo;
    }

    public List<AiObservabilityTrendVO> trends(AiObservabilityTrendsQuery query) {
        String dimension = query.getDimension();
        if (!List.of("model", "agent").contains(dimension)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "dimension 仅支持 model/agent");
        }
        List<AiObservabilityTrendVO> items = new ArrayList<>();
        for (AiTraceTrendRead row : observabilityMapper.selectTrends(dimension,
                AiDateTimeUtils.parse(query.getStartTime()), AiDateTimeUtils.parse(query.getEndTime()))) {
            AiObservabilityTrendVO item = new AiObservabilityTrendVO();
            if ("model".equals(dimension)) {
                item.setModel(row.getDimension());
            } else {
                item.setAgentCode(row.getDimension());
            }
            item.setDate(row.getDate());
            item.setCallCount(row.getCallCount());
            item.setSuccessCount(row.getSuccessCount());
            long callCount = row.getCallCount() == null ? 0 : row.getCallCount();
            long successCount = row.getSuccessCount() == null ? 0 : row.getSuccessCount();
            item.setSuccessRate(callCount == 0 ? 0.0
                    : BigDecimal.valueOf(successCount * 100.0)
                            .divide(BigDecimal.valueOf(callCount), 2, RoundingMode.HALF_UP).doubleValue());
            item.setAvgFirstTokenMs(scale2(row.getAvgFirstTokenMs()));
            item.setAvgDurationMs(scale2(row.getAvgDurationMs()));
            items.add(item);
        }
        return items;
    }

    /** 过程链导出（CSV，UTF-8 BOM 便于 Excel 打开），按检索条件全量导出并限行数 */
    public byte[] exportTraces(AiObservabilityTraceQuery query) {
        LocalDateTime start = AiDateTimeUtils.parse(query.getStartTime());
        LocalDateTime end = AiDateTimeUtils.parse(query.getEndTime());
        long count = observabilityMapper.countTraces(query, start, end);
        if (count > TaskConstants.MAX_ROWS) {
            throw new BusinessException(ResultCode.EXPORT_ROWS_EXCEED_LIMIT,
                    "导出行数 " + count + " 超出限制 " + TaskConstants.MAX_ROWS);
        }
        StringBuilder csv = new StringBuilder(String.join(",",
                "trace_id", "conversation_id", "message_id", "agent_code", "model", "status", "error_type",
                "duration_ms", "first_token_ms", "llm_call_count", "total_tokens", "prompt_tokens",
                "completion_tokens", "cached_tokens", "step_count", "create_time")).append('\n');
        for (SysAiTrace trace : observabilityMapper.selectTraces(query, start, end)) {
            csv.append(String.join(",",
                    text(trace.getTraceId()), text(trace.getConversationId()), text(trace.getMessageId()),
                    text(trace.getAgentCode()), text(trace.getModel()), text(trace.getStatus()),
                    text(trace.getErrorType()), text(trace.getDurationMs()), text(trace.getFirstTokenMs()),
                    text(trace.getLlmCallCount()), text(trace.getTotalTokens()), text(trace.getPromptTokens()),
                    text(trace.getCompletionTokens()), text(trace.getCachedTokens()), text(trace.getStepCount()),
                    trace.getCreateTime() == null ? "" : trace.getCreateTime().format(CSV_TIME))).append('\n');
        }
        byte[] payload = csv.toString().getBytes(StandardCharsets.UTF_8);
        byte[] withBom = new byte[payload.length + 3];
        withBom[0] = (byte) 0xEF;
        withBom[1] = (byte) 0xBB;
        withBom[2] = (byte) 0xBF;
        System.arraycopy(payload, 0, withBom, 3, payload.length);
        return withBom;
    }

    // ── 时间线 ──────────────────────────────────────────────

    private AiObservabilityTimelineVO buildTimeline(Long conversationId, boolean admin, boolean includeRaw) {
        SysAiConversation conv;
        if (admin) {
            conv = conversationMapper.selectById(conversationId);
        } else {
            conv = conversationMapper.selectOne(new LambdaQueryWrapper<SysAiConversation>()
                    .eq(SysAiConversation::getId, conversationId)
                    .eq(SysAiConversation::getUserId, SecurityUtils.getUserId()));
        }
        if (conv == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在");
        }

        // 沿当前激活分支链取全量消息（分支对话时间序与链序可能不一致，以链为准）
        Long chainStart = conv.getCurrentBranchMessageId() != null
                ? conv.getCurrentBranchMessageId() : messageMapper.getLastMessageId(conversationId);
        List<SysAiMessage> chain = chainByTail(conversationId, chainStart);
        List<Round> rounds = splitRounds(chain);
        List<SysAiTrace> traces = traceMapper.selectList(new LambdaQueryWrapper<SysAiTrace>()
                .eq(SysAiTrace::getConversationId, conversationId)
                .orderByAsc(SysAiTrace::getCreateTime)
                .orderByAsc(SysAiTrace::getId));
        attachTraces(rounds, traces);

        Map<String, List<SysAiLlmCall>> callsMap = callsByTraces(traces.stream().map(SysAiTrace::getTraceId).toList());
        List<Long> assistantIds = rounds.stream().map(round -> round.assistant).filter(Objects::nonNull)
                .map(SysAiMessage::getId).toList();
        Map<Long, List<SysAiAgentThought>> thoughtsMap = thoughtsByMessages(assistantIds);

        Map<String, List<SysAiBilling>> billingByRequest = new HashMap<>();
        Map<Long, List<SysAiBilling>> billingByMessage = new HashMap<>();
        for (SysAiBilling billing : billingMapper.selectList(new LambdaQueryWrapper<SysAiBilling>()
                .eq(SysAiBilling::getConversationId, conversationId)
                .orderByAsc(SysAiBilling::getId))) {
            if (billing.getRequestId() != null && !billing.getRequestId().isBlank()) {
                billingByRequest.computeIfAbsent(billing.getRequestId(), k -> new ArrayList<>()).add(billing);
            } else {
                billingByMessage.computeIfAbsent(billing.getMessageId(), k -> new ArrayList<>()).add(billing);
            }
        }

        List<AiObservabilityTimelineVO.Round> timelineRounds = new ArrayList<>();
        for (Round round : rounds) {
            List<AiObservabilityTimelineVO.Trace> timelineTraces = new ArrayList<>();
            for (int idx = 0; idx < round.traces.size(); idx++) {
                SysAiTrace trace = round.traces.get(idx);
                boolean isPrimary = idx == 0 && "conversation".equals(trace.getTraceType());
                List<SysAiBilling> billing = billingByRequest.get(trace.getTraceId());
                if ((billing == null || billing.isEmpty()) && isPrimary && round.assistant != null) {
                    billing = billingByMessage.getOrDefault(round.assistant.getId(), List.of());
                }
                AiObservabilityTimelineVO.Trace item = new AiObservabilityTimelineVO.Trace();
                item.setTraceId(trace.getTraceId());
                item.setTraceType(trace.getTraceType());
                item.setStatus(trace.getStatus());
                item.setErrorType(trace.getErrorType());
                item.setErrorDetail(readJson(trace.getErrorDetail()));
                item.setModel(trace.getModel());
                item.setDurationMs(trace.getDurationMs());
                item.setCreateTime(trace.getCreateTime());
                item.setEvents(traceEvents(trace,
                        callsMap.getOrDefault(trace.getTraceId(), List.of()),
                        isPrimary && round.assistant != null
                                ? thoughtsMap.getOrDefault(round.assistant.getId(), List.of()) : List.of(),
                        billing == null ? List.of() : billing,
                        isPrimary ? round.user : null,
                        includeRaw));
                timelineTraces.add(item);
            }
            AiObservabilityTimelineVO.Round roundVO = new AiObservabilityTimelineVO.Round();
            roundVO.setUserMessage(round.user == null ? null : toTimelineMessage(round.user));
            roundVO.setAssistantMessage(round.assistant == null ? null : toTimelineMessage(round.assistant));
            roundVO.setTraces(timelineTraces);
            timelineRounds.add(roundVO);
        }

        AiObservabilityTimelineVO.Conversation conversation = new AiObservabilityTimelineVO.Conversation();
        conversation.setId(conv.getId());
        conversation.setTitle(conv.getTitle());
        conversation.setUserId(conv.getUserId());
        conversation.setAgentCode(conv.getAgentCode());
        conversation.setCreateTime(conv.getCreateTime());

        AiObservabilityTimelineVO vo = new AiObservabilityTimelineVO();
        vo.setConversation(conversation);
        vo.setRounds(timelineRounds);
        return vo;
    }

    private record Round(SysAiMessage user, SysAiMessage assistant, List<SysAiTrace> traces) {
    }

    /** 消息链切轮次：user 开新轮，assistant 归属当前轮（链首 assistant 单独成轮） */
    private List<Round> splitRounds(List<SysAiMessage> chain) {
        List<Round> rounds = new ArrayList<>();
        for (SysAiMessage msg : chain) {
            if ("user".equals(msg.getRole()) || rounds.isEmpty()) {
                rounds.add(new Round("user".equals(msg.getRole()) ? msg : null, null, new ArrayList<>()));
            }
            if ("assistant".equals(msg.getRole())) {
                Round current = rounds.get(rounds.size() - 1);
                rounds.set(rounds.size() - 1, new Round(current.user, msg, current.traces));
            }
        }
        return rounds;
    }

    /**
     * trace 归属轮次：有 message_id 按消息配对（resume 多 trace 并列同一轮）；无 message_id
     * （summary/memory_extraction 旁路）挂触发时点最新轮次；主对话 trace 在前、旁路在后
     */
    private void attachTraces(List<Round> rounds, List<SysAiTrace> traces) {
        for (SysAiTrace trace : traces) {
            Round target = null;
            if (trace.getMessageId() != null) {
                for (Round round : rounds) {
                    if (Objects.equals(round.user == null ? null : round.user.getId(), trace.getMessageId())
                            || Objects.equals(round.assistant == null ? null : round.assistant.getId(),
                            trace.getMessageId())) {
                        target = round;
                        break;
                    }
                }
            }
            if (target == null) {
                // 轮次按触发时间正序，取 create_time 之前（含）的最近一轮
                for (Round round : rounds) {
                    SysAiMessage anchor = round.user != null ? round.user : round.assistant;
                    if (anchor != null && !anchor.getCreateTime().isAfter(trace.getCreateTime())) {
                        target = round;
                    } else {
                        break;
                    }
                }
                if (target == null && !rounds.isEmpty()) {
                    target = rounds.get(0);
                }
            }
            if (target != null) {
                target.traces().add(trace);
            }
        }
        for (Round round : rounds) {
            round.traces().sort(Comparator
                    .comparing((SysAiTrace t) -> !"conversation".equals(t.getTraceType()))
                    .thenComparing(SysAiTrace::getCreateTime)
                    .thenComparing(SysAiTrace::getId));
        }
    }

    private List<AiObservabilityTimelineVO.Event> traceEvents(SysAiTrace trace, List<SysAiLlmCall> calls,
                                                              List<SysAiAgentThought> thoughts,
                                                              List<SysAiBilling> billingRows,
                                                              SysAiMessage userMessage, boolean includeRaw) {
        List<AiObservabilityTimelineVO.Event> events = new ArrayList<>();
        Object snapshot = readJson(trace.getContextSnapshot());
        LocalDateTime approxStart = trace.getDurationMs() == null ? trace.getCreateTime()
                : trace.getCreateTime().minusNanos(trace.getDurationMs() * 1_000_000L);
        AiObservabilityTimelineVO.Event contextEvent = new AiObservabilityTimelineVO.Event();
        contextEvent.setKind("context");
        contextEvent.setTs(approxStart);
        contextEvent.setSnapshot(snapshot);
        events.add(contextEvent);
        if (snapshot instanceof Map<?, ?> map && map.get("events") instanceof List<?> items) {
            for (Object item : items) {
                AiObservabilityTimelineVO.Event event = new AiObservabilityTimelineVO.Event();
                event.setKind("system_event");
                event.setTs(approxStart);
                event.setEvent(item instanceof Map<?, ?> m ? text(m.get("event")) : null);
                event.setDetail(item);
                events.add(event);
            }
        }
        for (SysAiLlmCall call : calls) {
            AiObservabilityTimelineVO.Event event = new AiObservabilityTimelineVO.Event();
            event.setKind("llm_call");
            event.setTs(call.getStartTime());
            event.setSeq(call.getSeq());
            event.setModel(call.getModel());
            event.setStatus(call.getStatus());
            event.setDurationMs(call.getDurationMs());
            event.setFirstTokenMs(call.getFirstTokenMs());
            event.setPromptTokens(call.getPromptTokens());
            event.setCompletionTokens(call.getCompletionTokens());
            event.setCachedTokens(call.getCachedTokens());
            event.setToolCall(readJson(call.getToolCall()));
            event.setAttempts(readJson(call.getAttempts()));
            event.setRawRequest(includeRaw ? readJson(call.getRawRequest()) : null);
            event.setRawResponse(includeRaw ? readJson(call.getRawResponse()) : null);
            // summary 恒为纯计数单形状（消息/工具全文唯一通道走 rawRequest），无全文兜底形状
            Map<String, Object> summary = new LinkedHashMap<>();
            summary.put("inputSnapshot", slimInputSnapshot(readJson(call.getInputSnapshot())));
            summary.put("outputSnapshot", readJson(call.getOutputSnapshot()));
            event.setSummary(summary);
            events.add(event);
        }
        for (SysAiAgentThought thought : thoughts) {
            AiObservabilityTimelineVO.Event event = new AiObservabilityTimelineVO.Event();
            event.setKind("tool_exec");
            event.setTs(thought.getCreateTime());
            event.setPosition(thought.getPosition());
            event.setTool(thought.getTool());
            event.setThought(thought.getThought());
            event.setToolInput(thought.getToolInput());
            event.setObservation(thought.getObservation());
            event.setStatus(thought.getStatus());
            event.setLatencyMs(thought.getLatencyMs());
            event.setAgentCode(thought.getAgentCode());
            event.setIsSubagent(thought.getIsSubagent());
            events.add(event);
        }
        for (SysAiBilling billing : billingRows) {
            AiObservabilityTimelineVO.Event event = new AiObservabilityTimelineVO.Event();
            event.setKind("billing");
            event.setTs(billing.getCreateTime());
            event.setBillType(billing.getBillType());
            event.setCredits(billing.getCredits());
            Map<String, Object> tokens = new LinkedHashMap<>();
            tokens.put("input", billing.getInputTokens());
            tokens.put("output", billing.getOutputTokens());
            tokens.put("cached", billing.getCachedInputTokens());
            event.setTokens(tokens);
            events.add(event);
        }
        if (userMessage != null) {
            AiObservabilityTimelineVO.Event event = new AiObservabilityTimelineVO.Event();
            event.setKind("input");
            event.setTs(userMessage.getCreateTime());
            event.setMessage(toTimelineMessage(userMessage));
            events.add(event);
        }
        events.sort(Comparator
                .comparing((AiObservabilityTimelineVO.Event e) -> e.getTs() == null)
                .thenComparing(e -> e.getTs() == null ? LocalDateTime.MIN : e.getTs())
                .thenComparing(e -> EVENT_PRIORITY.getOrDefault(e.getKind(), 9))
                .thenComparing(e -> e.getSeq() == null ? 0 : e.getSeq())
                .thenComparing(e -> e.getPosition() == null ? 0 : e.getPosition()));
        return events;
    }

    /**
     * 输入快照瘦身：保留按角色计数/token 估算/工具数/用户/系统提示 token 数，
     * 去掉 messages.items 全文与 tools 定义清单（全文唯一通道走 rawRequest）
     */
    private Object slimInputSnapshot(Object snapshot) {
        if (!(snapshot instanceof Map<?, ?> map)) {
            return null;
        }
        Map<String, Object> slim = new LinkedHashMap<>();
        if (map.get("messages") instanceof Map<?, ?> messages) {
            Map<String, Object> countsOnly = new LinkedHashMap<>();
            for (Map.Entry<?, ?> entry : messages.entrySet()) {
                if (!"items".equals(entry.getKey())) {
                    countsOnly.put(String.valueOf(entry.getKey()), entry.getValue());
                }
            }
            slim.put("messages", countsOnly);
        }
        for (String key : List.of("system_tokens", "tool_count", "user_id")) {
            if (map.containsKey(key)) {
                slim.put(key, map.get(key));
            }
        }
        return slim;
    }

    // ── 明细查询 ────────────────────────────────────────────

    private List<AiObservabilityTraceDetailVO.LlmCall> llmCallsByTrace(String traceId) {
        List<AiObservabilityTraceDetailVO.LlmCall> items = new ArrayList<>();
        for (SysAiLlmCall call : llmCallMapper.selectList(new LambdaQueryWrapper<SysAiLlmCall>()
                .eq(SysAiLlmCall::getTraceId, traceId)
                .orderByAsc(SysAiLlmCall::getSeq))) {
            AiObservabilityTraceDetailVO.LlmCall item = new AiObservabilityTraceDetailVO.LlmCall();
            item.setSeq(call.getSeq());
            item.setStepPosition(call.getStepPosition());
            item.setModel(call.getModel());
            item.setStatus(call.getStatus());
            item.setErrorType(call.getErrorType());
            item.setDurationMs(call.getDurationMs());
            item.setFirstTokenMs(call.getFirstTokenMs());
            item.setPromptTokens(call.getPromptTokens());
            item.setCompletionTokens(call.getCompletionTokens());
            item.setCachedTokens(call.getCachedTokens());
            item.setToolCall(readJson(call.getToolCall()));
            item.setInputSnapshot(readJson(call.getInputSnapshot()));
            item.setOutputSnapshot(readJson(call.getOutputSnapshot()));
            item.setAttempts(readJson(call.getAttempts()));
            item.setStartTime(call.getStartTime());
            item.setRawRequest(readJson(call.getRawRequest()));
            item.setRawResponse(readJson(call.getRawResponse()));
            item.setCreateTime(call.getCreateTime());
            items.add(item);
        }
        return items;
    }

    private Map<String, List<SysAiLlmCall>> callsByTraces(List<String> traceIds) {
        if (traceIds.isEmpty()) {
            return Map.of();
        }
        Map<String, List<SysAiLlmCall>> grouped = new HashMap<>();
        for (SysAiLlmCall call : llmCallMapper.selectList(new LambdaQueryWrapper<SysAiLlmCall>()
                .in(SysAiLlmCall::getTraceId, traceIds)
                .orderByAsc(SysAiLlmCall::getSeq))) {
            grouped.computeIfAbsent(call.getTraceId(), k -> new ArrayList<>()).add(call);
        }
        return grouped;
    }

    private List<AiAgentThoughtVO> thoughtsByMessage(Long messageId) {
        List<AiAgentThoughtVO> items = new ArrayList<>();
        for (SysAiAgentThought thought : thoughtMapper.selectList(new LambdaQueryWrapper<SysAiAgentThought>()
                .eq(SysAiAgentThought::getMessageId, messageId)
                .orderByAsc(SysAiAgentThought::getPosition))) {
            items.add(toThoughtVO(thought));
        }
        return items;
    }

    private Map<Long, List<SysAiAgentThought>> thoughtsByMessages(List<Long> messageIds) {
        if (messageIds.isEmpty()) {
            return Map.of();
        }
        Map<Long, List<SysAiAgentThought>> grouped = new HashMap<>();
        for (SysAiAgentThought thought : thoughtMapper.selectList(new LambdaQueryWrapper<SysAiAgentThought>()
                .in(SysAiAgentThought::getMessageId, messageIds)
                .orderByAsc(SysAiAgentThought::getPosition))) {
            grouped.computeIfAbsent(thought.getMessageId(), k -> new ArrayList<>()).add(thought);
        }
        return grouped;
    }

    private List<SysAiBilling> billingByRequestId(String traceId) {
        return billingMapper.selectList(new LambdaQueryWrapper<SysAiBilling>()
                .eq(SysAiBilling::getRequestId, traceId)
                .orderByAsc(SysAiBilling::getId));
    }

    private List<SysAiBilling> billingByMessage(Long messageId) {
        return billingMapper.selectList(new LambdaQueryWrapper<SysAiBilling>()
                .eq(SysAiBilling::getMessageId, messageId)
                .orderByAsc(SysAiBilling::getId));
    }

    private List<AiObservabilityTraceDetailVO.Artifact> artifactsByMessage(Long messageId) {
        List<AiObservabilityTraceDetailVO.Artifact> items = new ArrayList<>();
        for (SysAiArtifact artifact : artifactMapper.selectList(new LambdaQueryWrapper<SysAiArtifact>()
                .eq(SysAiArtifact::getMessageId, messageId)
                .orderByDesc(SysAiArtifact::getCreateTime)
                .orderByDesc(SysAiArtifact::getId))) {
            AiObservabilityTraceDetailVO.Artifact item = new AiObservabilityTraceDetailVO.Artifact();
            item.setId(artifact.getId());
            item.setType(artifact.getType());
            item.setSummary(artifact.getSummary());
            item.setRefType(artifact.getRefType());
            item.setRefId(artifact.getRefId());
            item.setCreateTime(artifact.getCreateTime());
            items.add(item);
        }
        return items;
    }

    private List<AiObservabilityTraceDetailVO.Message> messagesByConversation(Long conversationId) {
        List<AiObservabilityTraceDetailVO.Message> items = new ArrayList<>();
        for (SysAiMessage message : messageMapper.selectList(new LambdaQueryWrapper<SysAiMessage>()
                .eq(SysAiMessage::getConversationId, conversationId)
                .orderByAsc(SysAiMessage::getCreateTime)
                .orderByAsc(SysAiMessage::getId)
                .last("LIMIT " + TRACE_MESSAGE_LIMIT))) {
            AiObservabilityTraceDetailVO.Message item = new AiObservabilityTraceDetailVO.Message();
            item.setId(message.getId());
            item.setConversationId(message.getConversationId());
            item.setParentMessageId(message.getParentMessageId());
            item.setRole(message.getRole());
            item.setContent(message.getContent());
            item.setStatus(message.getStatus());
            item.setModel(message.getModel());
            item.setInputTokens(message.getInputTokens());
            item.setOutputTokens(message.getOutputTokens());
            item.setCreateTime(message.getCreateTime());
            items.add(item);
        }
        return items;
    }

    private Map<Long, String> conversationTitles(List<SysAiTrace> traces) {
        if (traces.isEmpty()) {
            return Map.of();
        }
        Map<Long, String> titles = new HashMap<>();
        for (SysAiConversation conversation : conversationMapper.selectList(
                new LambdaQueryWrapper<SysAiConversation>()
                        .select(SysAiConversation::getId, SysAiConversation::getTitle)
                        .in(SysAiConversation::getId,
                                traces.stream().map(SysAiTrace::getConversationId).distinct().toList()))) {
            titles.put(conversation.getId(), conversation.getTitle());
        }
        return titles;
    }

    private Long conversationOwner(Long conversationId) {
        SysAiConversation conv = conversationMapper.selectById(conversationId);
        return conv == null ? null : conv.getUserId();
    }

    /** 按当前激活分支末端回溯完整消息链（一次查询本会话消息 + 内存组链，visited 防环） */
    private List<SysAiMessage> chainByTail(Long conversationId, Long tailMessageId) {
        if (tailMessageId == null) {
            return List.of();
        }
        Map<Long, SysAiMessage> byId = new LinkedHashMap<>();
        for (SysAiMessage message : messageMapper.selectList(new LambdaQueryWrapper<SysAiMessage>()
                .eq(SysAiMessage::getConversationId, conversationId)
                .orderByAsc(SysAiMessage::getId))) {
            byId.putIfAbsent(message.getId(), message);
        }
        List<SysAiMessage> chain = new ArrayList<>();
        Set<Long> visited = new HashSet<>();
        Long cursor = tailMessageId;
        while (cursor != null && visited.add(cursor)) {
            SysAiMessage message = byId.get(cursor);
            if (message == null) {
                break;
            }
            chain.add(message);
            cursor = message.getParentMessageId();
        }
        Collections.reverse(chain);
        return chain;
    }

    private boolean isAudit() {
        return SecurityUtils.isRoot() || SecurityUtils.getPerms().contains(AUDIT_PERMISSION);
    }

    // ── 映射 ────────────────────────────────────────────────

    private AiObservabilityTraceItemVO toItemVO(SysAiTrace trace) {
        AiObservabilityTraceItemVO vo = new AiObservabilityTraceItemVO();
        copyTraceFields(trace, vo);
        return vo;
    }

    private void copyTraceFields(SysAiTrace trace, AiObservabilityTraceItemVO vo) {
        vo.setTraceId(trace.getTraceId());
        vo.setConversationId(trace.getConversationId());
        vo.setMessageId(trace.getMessageId());
        vo.setAgentCode(trace.getAgentCode());
        vo.setTraceType(trace.getTraceType());
        vo.setModel(trace.getModel());
        vo.setStatus(trace.getStatus());
        vo.setErrorType(trace.getErrorType());
        vo.setDurationMs(trace.getDurationMs());
        vo.setFirstTokenMs(trace.getFirstTokenMs());
        vo.setLlmCallCount(trace.getLlmCallCount());
        vo.setTotalTokens(trace.getTotalTokens());
        vo.setPromptTokens(trace.getPromptTokens());
        vo.setCompletionTokens(trace.getCompletionTokens());
        vo.setCachedTokens(trace.getCachedTokens());
        vo.setStepCount(trace.getStepCount());
        vo.setCreateTime(trace.getCreateTime());
    }

    private void copyTraceFields(SysAiTrace trace, AiObservabilityTraceDetailVO vo) {
        copyTraceFields(trace, (AiObservabilityTraceItemVO) vo);
    }

    private AiObservabilityTimelineVO.Message toTimelineMessage(SysAiMessage message) {
        AiObservabilityTimelineVO.Message vo = new AiObservabilityTimelineVO.Message();
        vo.setId(message.getId());
        vo.setRole(message.getRole());
        vo.setContent(message.getContent());
        vo.setStatus(message.getStatus());
        vo.setModel(message.getModel());
        vo.setInputTokens(message.getInputTokens());
        vo.setOutputTokens(message.getOutputTokens());
        vo.setCreateTime(message.getCreateTime());
        return vo;
    }

    private AiObservabilityTraceDetailVO.Billing toBillingVO(SysAiBilling billing) {
        AiObservabilityTraceDetailVO.Billing vo = new AiObservabilityTraceDetailVO.Billing();
        vo.setBillType(billing.getBillType());
        vo.setModel(billing.getModel());
        vo.setActualModel(billing.getActualModel());
        vo.setProviderId(billing.getProviderId());
        vo.setInputTokens(billing.getInputTokens());
        vo.setOutputTokens(billing.getOutputTokens());
        vo.setCachedInputTokens(billing.getCachedInputTokens());
        vo.setCredits(billing.getCredits());
        vo.setCreditsSaved(billing.getCreditsSaved());
        vo.setErrorCode(billing.getErrorCode());
        vo.setLatencyMs(billing.getLatencyMs());
        vo.setRequestId(billing.getRequestId());
        vo.setCreateTime(billing.getCreateTime());
        return vo;
    }

    private AiAgentThoughtVO toThoughtVO(SysAiAgentThought thought) {
        AiAgentThoughtVO vo = new AiAgentThoughtVO();
        vo.setId(thought.getId());
        vo.setMessageId(thought.getMessageId());
        vo.setConversationId(thought.getConversationId());
        vo.setPosition(thought.getPosition());
        vo.setAgentCode(thought.getAgentCode());
        vo.setIsSubagent(thought.getIsSubagent());
        vo.setThought(thought.getThought());
        vo.setTool(thought.getTool());
        vo.setToolInput(thought.getToolInput());
        vo.setObservation(thought.getObservation());
        vo.setStatus(thought.getStatus());
        vo.setLatencyMs(thought.getLatencyMs());
        vo.setError(thought.getError());
        vo.setCreateTime(thought.getCreateTime());
        return vo;
    }

    private Object readJson(String json) {
        if (json == null || json.isBlank()) {
            return null;
        }
        try {
            return objectMapper.readValue(json, Object.class);
        } catch (Exception e) {
            log.warn("可观测性 JSON 列解析失败，按空值透出: {}", json.length() > 64 ? json.substring(0, 64) : json);
            return null;
        }
    }

    private static Double scale2(BigDecimal value) {
        return value == null ? null : value.setScale(2, RoundingMode.HALF_UP).doubleValue();
    }

    private static String text(Object value) {
        return value == null ? "" : String.valueOf(value);
    }

    private static Long parseLong(String value) {
        try {
            return Long.parseLong(value);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private static <T> IPage<T> pageOf(long pageNum, long pageSize, List<T> records, long total) {
        Page<T> page = new Page<>(pageNum, pageSize, total);
        page.setRecords(new ArrayList<>(records));
        return page;
    }
}

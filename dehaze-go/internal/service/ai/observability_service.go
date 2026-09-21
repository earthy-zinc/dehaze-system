package ai

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	aidomainrepo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	"github.com/earthyzinc/dehaze-go/pkg/common"
)

// 可观测性查询口径常量（与 dehaze-python app/service/ai_observability_service.py 一致）。
const (
	// observabilityMaxRows 过程链导出上限，超出报 A0709。
	observabilityMaxRows = 100000
	// timelineMessageLimit 过程链详情回放的消息条数上限（详情的 messages 字段）。
	timelineMessageLimit = 1000
)

// timelineEventPriority 同刻事件的业务序（input → context → system_event → llm_call → tool_exec → billing）。
var timelineEventPriority = map[string]int{
	"input": 0, "context": 1, "system_event": 2, "llm_call": 3, "tool_exec": 4, "billing": 5,
}

// ObservabilityService AI 可观测性查询（F-M08-013）：过程链检索/详情/导出、会话审计时间线、
// 异常总览、资源消耗聚合与性能趋势。聚合基于 sys_ai_trace 现有字段直查，不引入额外存储。
type ObservabilityService struct {
	repo        *airepo.ObservabilityRepository
	billingRepo *airepo.BillingRepository
	convs       *aidomainrepo.ConversationRepository
	messages    *aidomainrepo.MessageRepository
	thoughts    *aidomainrepo.ThoughtRepository
}

func NewObservabilityService(
	repo *airepo.ObservabilityRepository,
	billingRepo *airepo.BillingRepository,
	convs *aidomainrepo.ConversationRepository,
	messages *aidomainrepo.MessageRepository,
	thoughts *aidomainrepo.ThoughtRepository,
) *ObservabilityService {
	return &ObservabilityService{
		repo: repo, billingRepo: billingRepo, convs: convs,
		messages: messages, thoughts: thoughts,
	}
}

// ==================== 异常总览 ====================

// Summary 异常总览统计：状态分布 + 配额拒绝 + 高风险调用。
func (s *ObservabilityService) Summary(ctx context.Context) (*vo.ObservabilitySummaryVO, error) {
	counts, err := s.repo.TraceStatusCount(ctx)
	if err != nil {
		return nil, err
	}
	quotaRejected, err := s.repo.CountQuotaRejected(ctx)
	if err != nil {
		return nil, err
	}
	highRisk, err := s.repo.CountHighRisk(ctx)
	if err != nil {
		return nil, err
	}
	var total int64
	for _, count := range counts {
		total += count
	}
	return &vo.ObservabilitySummaryVO{
		Total:            total,
		SuccessCount:     counts[1],
		FailedCount:      counts[2],
		InterruptedCount: counts[3],
		TimeoutCount:     counts[4],
		QuotaRejected:    quotaRejected,
		HighRiskCalls:    highRisk,
	}, nil
}

// ==================== 过程链检索 ====================

// ListTraces 过程链分页检索（会话/用户/状态/智能体/模型/失败类型/关键词/能力维度/时间）。
func (s *ObservabilityService) ListTraces(ctx context.Context, q *bo.TracePageQuery) (*vo.PageResult[vo.TraceVO], error) {
	page, size := q.PageNum, q.PageSize
	start, end, err := observabilityRange(q.StartTime, q.EndTime)
	if err != nil {
		return nil, err
	}
	if err := validateTraceQuery(q); err != nil {
		return nil, err
	}
	items, total, err := s.repo.PaginateTraces(ctx, airepo.FilterFromQuery(q, start, end), page, size)
	if err != nil {
		return nil, err
	}
	titles, err := s.conversationTitles(ctx, items)
	if err != nil {
		return nil, err
	}
	list := make([]vo.TraceVO, 0, len(items))
	for i := range items {
		row := traceToVO(&items[i])
		row.ConversationTitle = titles[items[i].ConversationID]
		list = append(list, row)
	}
	return &vo.PageResult[vo.TraceVO]{List: list, Total: total}, nil
}

// conversationTitles 会话标题批量回填（检索行透出会话归属，供前端跳转会话时间线）。
func (s *ObservabilityService) conversationTitles(ctx context.Context, traces []model.SysAiTrace) (map[int64]*string, error) {
	ids := make([]int64, 0, len(traces))
	seen := make(map[int64]struct{}, len(traces))
	for i := range traces {
		if _, ok := seen[traces[i].ConversationID]; ok {
			continue
		}
		seen[traces[i].ConversationID] = struct{}{}
		ids = append(ids, traces[i].ConversationID)
	}
	raw, err := s.repo.ConversationTitles(ctx, ids)
	if err != nil {
		return nil, err
	}
	titles := make(map[int64]*string, len(raw))
	for id, title := range raw {
		value := title
		titles[id] = &value
	}
	return titles, nil
}

// GetTrace 过程链详情：上下文快照 + LLM 调用回放（seq 正序）+ 推理步骤 + 计费 + 产物 + 会话消息。
//
// 管理员可查全量；普通用户仅可查自己会话的过程链，跨会话访问与不存在一律 A0401，不暴露他人过程链存在性。
func (s *ObservabilityService) GetTrace(
	ctx context.Context, traceID string, userID int64, admin bool,
) (*vo.TraceDetailVO, error) {
	trace, err := s.repo.GetTraceByTraceID(ctx, traceID)
	if err != nil {
		return nil, err
	}
	if trace == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "过程链不存在")
	}
	if !admin {
		conv, convErr := s.convs.GetByID(ctx, trace.ConversationID)
		if convErr != nil {
			return nil, convErr
		}
		if conv == nil || conv.UserID != userID {
			return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "过程链不存在")
		}
	}

	detail := &vo.TraceDetailVO{
		TraceVO:         traceToVO(trace),
		ContextSnapshot: trace.ContextSnapshot,
		ErrorDetail:     trace.ErrorDetail,
		LlmCalls:        []vo.LlmCallVO{},
		Thoughts:        []vo.AgentThoughtVO{},
		Messages:        []vo.TraceMessageVO{},
		Billing:         []vo.TraceBillingVO{},
		Artifacts:       []vo.TraceArtifactVO{},
	}
	calls, err := s.repo.ListLLMCallsByTrace(ctx, traceID)
	if err != nil {
		return nil, err
	}
	for i := range calls {
		detail.LlmCalls = append(detail.LlmCalls, llmCallToVO(&calls[i]))
	}

	if trace.MessageID != nil {
		thoughts, thoughtErr := s.thoughts.ListByMessage(ctx, *trace.MessageID)
		if thoughtErr != nil {
			return nil, thoughtErr
		}
		for i := range thoughts {
			detail.Thoughts = append(detail.Thoughts, thoughtToVO(&thoughts[i]))
		}
		// 计费优先按 request_id=trace_id 关联（调用级精确归因），无命中回退 message_id
		billingRows, billingErr := s.billingRepo.ListBillingByRequestID(ctx, traceID)
		if billingErr != nil {
			return nil, billingErr
		}
		if len(billingRows) == 0 {
			billingRows, billingErr = s.billingRepo.ListBillingByMessage(ctx, *trace.MessageID)
			if billingErr != nil {
				return nil, billingErr
			}
		}
		for i := range billingRows {
			detail.Billing = append(detail.Billing, billingToVO(&billingRows[i]))
		}
		artifacts, artifactErr := s.repo.ListArtifactsByMessage(ctx, *trace.MessageID)
		if artifactErr != nil {
			return nil, artifactErr
		}
		for i := range artifacts {
			detail.Artifacts = append(detail.Artifacts, artifactToVO(&artifacts[i]))
		}
	}

	messages, err := s.repo.ListMessagesAsc(ctx, trace.ConversationID, timelineMessageLimit)
	if err != nil {
		return nil, err
	}
	for i := range messages {
		detail.Messages = append(detail.Messages, traceMessageToVO(&messages[i]))
	}
	return detail, nil
}

// ==================== 会话审计时间线 ====================

// timelineRound 消息链切分出的轮次（user 开新轮，assistant 归属当前轮）。
type timelineRound struct {
	user      *model.SysAiMessage
	assistant *model.SysAiMessage
	traces    []model.SysAiTrace
}

// ConversationTimeline 会话审计时间线：轮次（user→assistant 配对）+ 轮内事件按 ts 交织。
//
// 管理员可查全量；普通用户仅可查自己会话，跨会话/不存在一律 A0401。includeRaw=false 省略原始报文。
func (s *ObservabilityService) ConversationTimeline(
	ctx context.Context, conversationID, userID int64, admin, includeRaw bool,
) (*vo.TimelineVO, error) {
	var conv *model.SysAiConversation
	var err error
	if admin {
		conv, err = s.convs.GetByID(ctx, conversationID)
	} else {
		conv, err = s.convs.GetByIDAndUser(ctx, conversationID, userID)
	}
	if err != nil {
		return nil, err
	}
	if conv == nil {
		return nil, common.NewBizError(common.RESOURCE_NOT_FOUND, "会话不存在")
	}

	// 沿当前激活分支链取全量消息（分支对话时间序与链序可能不一致，以链为准）
	chainStart := conv.CurrentBranchMessageID
	if chainStart == nil {
		if chainStart, err = s.messages.GetLastMessageID(ctx, conversationID); err != nil {
			return nil, err
		}
	}
	var chain []model.SysAiMessage
	if chainStart != nil {
		if chain, err = s.messages.GetChainByID(ctx, conversationID, *chainStart); err != nil {
			return nil, err
		}
	}
	rounds := splitTimelineRounds(chain)

	traces, err := s.repo.ListTracesByConversation(ctx, conversationID)
	if err != nil {
		return nil, err
	}
	attachTracesToRounds(rounds, traces)

	traceIDs := make([]string, 0, len(traces))
	for i := range traces {
		traceIDs = append(traceIDs, traces[i].TraceID)
	}
	callsByTrace, err := s.repo.ListLLMCallsByTraces(ctx, traceIDs)
	if err != nil {
		return nil, err
	}

	assistantIDs := make([]int64, 0, len(rounds))
	for _, round := range rounds {
		if round.assistant != nil {
			assistantIDs = append(assistantIDs, round.assistant.ID)
		}
	}
	thoughtsByMessage, err := s.thoughts.ListByMessages(ctx, assistantIDs)
	if err != nil {
		return nil, err
	}

	// 计费按 request_id=trace_id 精确归因，无 request_id 的按 message_id 挂到该轮主对话 trace
	billingRows, err := s.billingRepo.ListBillingByConversation(ctx, conversationID)
	if err != nil {
		return nil, err
	}
	billingByRequest := make(map[string][]model.SysAiBilling)
	billingByMessage := make(map[int64][]model.SysAiBilling)
	for i := range billingRows {
		row := &billingRows[i]
		if row.RequestID != nil && *row.RequestID != "" {
			billingByRequest[*row.RequestID] = append(billingByRequest[*row.RequestID], *row)
		} else if row.MessageID != nil {
			billingByMessage[*row.MessageID] = append(billingByMessage[*row.MessageID], *row)
		}
	}

	result := &vo.TimelineVO{
		Conversation: vo.TimelineConversationVO{
			ID: conv.ID, Title: conv.Title, UserID: conv.UserID,
			AgentCode: conv.AgentCode, CreateTime: &conv.CreateTime,
		},
		Rounds: make([]vo.TimelineRoundVO, 0, len(rounds)),
	}
	for _, round := range rounds {
		timelineTraces := make([]vo.TimelineTraceVO, 0, len(round.traces))
		for idx := range round.traces {
			trace := &round.traces[idx]
			isPrimary := idx == 0 && trace.TraceType == "conversation"
			billing := billingByRequest[trace.TraceID]
			if len(billing) == 0 && isPrimary && round.assistant != nil {
				billing = billingByMessage[round.assistant.ID]
			}
			var thoughts []model.SysAiAgentThought
			if isPrimary && round.assistant != nil {
				thoughts = thoughtsByMessage[round.assistant.ID]
			}
			var userMessage *model.SysAiMessage
			if isPrimary {
				userMessage = round.user
			}
			timelineTraces = append(timelineTraces, vo.TimelineTraceVO{
				TraceID: trace.TraceID, TraceType: trace.TraceType, Status: trace.Status,
				ErrorType: trace.ErrorType, ErrorDetail: trace.ErrorDetail, Model: trace.Model,
				DurationMs: trace.DurationMs, CreateTime: trace.CreateTime,
				Events: s.traceEvents(trace, callsByTrace[trace.TraceID], thoughts, billing, userMessage, includeRaw),
			})
		}
		result.Rounds = append(result.Rounds, vo.TimelineRoundVO{
			UserMessage:      timelineMessageToVO(round.user),
			AssistantMessage: timelineMessageToVO(round.assistant),
			Traces:           timelineTraces,
		})
	}
	return result, nil
}

// ExportTimeline 会话时间线整体导出（JSON 全量含 raw 原始报文），复用管理端导出权限口径。
func (s *ObservabilityService) ExportTimeline(ctx context.Context, conversationID int64) (*vo.TimelineVO, error) {
	return s.ConversationTimeline(ctx, conversationID, 0, true, true)
}

// splitTimelineRounds 消息链切轮次：user 开新轮，assistant 归属当前轮（链首 assistant 单独成轮）。
func splitTimelineRounds(chain []model.SysAiMessage) []*timelineRound {
	rounds := make([]*timelineRound, 0, len(chain))
	for i := range chain {
		msg := &chain[i]
		if msg.Role == "user" || len(rounds) == 0 {
			round := &timelineRound{}
			if msg.Role == "user" {
				round.user = msg
			}
			rounds = append(rounds, round)
		}
		if msg.Role == "assistant" {
			rounds[len(rounds)-1].assistant = msg
		}
	}
	return rounds
}

// attachTracesToRounds 过程链归属轮次：有 message_id 按消息配对（resume 多 trace 并列同一轮）；
// 无 message_id（摘要/记忆提取旁路）挂触发时点最新轮次；主对话在前、旁路在后。
func attachTracesToRounds(rounds []*timelineRound, traces []model.SysAiTrace) {
	for i := range traces {
		trace := &traces[i]
		var target *timelineRound
		if trace.MessageID != nil {
			for _, round := range rounds {
				if round.user != nil && round.user.ID == *trace.MessageID {
					target = round
					break
				}
				if round.assistant != nil && round.assistant.ID == *trace.MessageID {
					target = round
					break
				}
			}
		}
		if target == nil {
			// 轮次按触发时间正序，取 create_time 之前（含）的最近一轮
			for _, round := range rounds {
				trigger := roundTriggerTime(round)
				if trigger == nil || !trigger.After(trace.CreateTime) {
					target = round
				} else {
					break
				}
			}
			if target == nil && len(rounds) > 0 {
				target = rounds[0]
			}
		}
		if target != nil {
			target.traces = append(target.traces, *trace)
		}
	}
	for _, round := range rounds {
		sort.SliceStable(round.traces, func(i, j int) bool {
			a, b := round.traces[i], round.traces[j]
			aPrimary := a.TraceType == "conversation"
			bPrimary := b.TraceType == "conversation"
			if aPrimary != bPrimary {
				return aPrimary
			}
			if !a.CreateTime.Equal(b.CreateTime) {
				return a.CreateTime.Before(b.CreateTime)
			}
			return a.ID < b.ID
		})
	}
}

func roundTriggerTime(round *timelineRound) *time.Time {
	if round.user != nil {
		return &round.user.CreateTime
	}
	if round.assistant != nil {
		return &round.assistant.CreateTime
	}
	return nil
}

// traceEvents 织入单条过程链的事件流：input/context/system_event/llm_call/tool_exec/billing。
//
// ts 锚点：llm_call 用 start_time（无值沉底），thought/billing 用 create_time，
// context/system_event 用 create_time-duration 近似（trace 表无精确起始时刻字段）。
func (s *ObservabilityService) traceEvents(
	trace *model.SysAiTrace, calls []model.SysAiLlmCall, thoughts []model.SysAiAgentThought,
	billing []model.SysAiBilling, userMessage *model.SysAiMessage, includeRaw bool,
) []vo.TimelineEventVO {
	approxStart := trace.CreateTime.Add(-time.Duration(trace.DurationMs) * time.Millisecond)
	events := make([]vo.TimelineEventVO, 0, len(calls)+len(thoughts)+len(billing)+2)
	events = append(events, vo.TimelineEventVO{
		Kind: "context", Ts: &approxStart, Snapshot: trace.ContextSnapshot,
	})
	for _, detail := range snapshotEvents(trace.ContextSnapshot) {
		events = append(events, vo.TimelineEventVO{
			Kind: "system_event", Ts: &approxStart, Event: snapshotEventName(detail), Detail: detail,
		})
	}
	for i := range calls {
		call := &calls[i]
		event := vo.TimelineEventVO{
			Kind: "llm_call", Ts: call.StartTime, Seq: &call.Seq, Model: call.Model,
			Status: &call.Status, DurationMs: &call.DurationMs, FirstTokenMs: call.FirstTokenMs,
			PromptTokens: &call.PromptTokens, CompletionTokens: &call.CompletionTokens,
			CachedTokens: &call.CachedTokens, ToolCall: call.ToolCall, Attempts: call.Attempts,
			// summary 恒为纯计数单形状（消息/工具全文唯一通道走 rawRequest）
			Summary: buildCallSummary(call.InputSnapshot, call.OutputSnapshot),
		}
		if includeRaw {
			event.RawRequest = call.RawRequest
			event.RawResponse = call.RawResponse
		}
		events = append(events, event)
	}
	for i := range thoughts {
		thought := &thoughts[i]
		latency := 0
		if thought.LatencyMs != nil {
			latency = *thought.LatencyMs
		}
		created := thought.CreateTime
		events = append(events, vo.TimelineEventVO{
			Kind: "tool_exec", Ts: &created, Position: &thought.Position, Tool: thought.Tool,
			Thought: thought.Thought, ToolInput: json.RawMessage(thought.ToolInput),
			Observation: thought.Observation, Status: &thought.Status, LatencyMs: &latency,
			AgentCode: thought.AgentCode, IsSubagent: &thought.IsSubagent,
		})
	}
	for i := range billing {
		row := &billing[i]
		created := row.CreateTime
		tokens, marshalErr := json.Marshal(map[string]int{
			"input": row.InputTokens, "output": row.OutputTokens, "cached": row.CachedInputTokens,
		})
		if marshalErr != nil {
			tokens = nil
		}
		events = append(events, vo.TimelineEventVO{
			Kind: "billing", Ts: &created, BillType: &row.BillType, Credits: &row.Credits,
			Tokens: tokens,
		})
	}
	if userMessage != nil {
		created := userMessage.CreateTime
		events = append(events, vo.TimelineEventVO{
			Kind: "input", Ts: &created, Message: timelineMessageToVO(userMessage),
		})
	}

	sort.SliceStable(events, func(i, j int) bool {
		a, b := events[i], events[j]
		// ts 缺失（start_time 为 NULL 的调用）沉底
		if (a.Ts == nil) != (b.Ts == nil) {
			return b.Ts == nil
		}
		if a.Ts != nil && !a.Ts.Equal(*b.Ts) {
			return a.Ts.Before(*b.Ts)
		}
		if timelineEventPriority[a.Kind] != timelineEventPriority[b.Kind] {
			return timelineEventPriority[a.Kind] < timelineEventPriority[b.Kind]
		}
		if eventInt(a.Seq) != eventInt(b.Seq) {
			return eventInt(a.Seq) < eventInt(b.Seq)
		}
		return eventInt(a.Position) < eventInt(b.Position)
	})
	return events
}

func eventInt(value *int) int {
	if value == nil {
		return 0
	}
	return *value
}

// snapshotEvents 快照内系统事件明细（context_snapshot.events，键保持写入原样）。
func snapshotEvents(snapshot json.RawMessage) []json.RawMessage {
	if len(snapshot) == 0 {
		return nil
	}
	var wrapper struct {
		Events []json.RawMessage `json:"events"`
	}
	if json.Unmarshal(snapshot, &wrapper) != nil {
		return nil
	}
	return wrapper.Events
}

func snapshotEventName(detail json.RawMessage) *string {
	var meta struct {
		Event *string `json:"event"`
	}
	if json.Unmarshal(detail, &meta) != nil {
		return nil
	}
	return meta.Event
}

// buildCallSummary 调用摘要：inputSnapshot 瘦身 + outputSnapshot（正文全文不在此通道）。
func buildCallSummary(input, output json.RawMessage) json.RawMessage {
	summary := make(map[string]json.RawMessage, 2)
	if slim := slimInputSnapshot(input); len(slim) > 0 {
		summary["inputSnapshot"] = slim
	}
	if len(output) > 0 {
		summary["outputSnapshot"] = output
	}
	if len(summary) == 0 {
		return nil
	}
	payload, err := json.Marshal(summary)
	if err != nil {
		return nil
	}
	return payload
}

// slimInputSnapshot 输入快照瘦身：保留按角色计数/token 估算/工具数/用户/系统提示 token 数，
// 去掉 messages.items 全文、system_content 与 tools 定义清单（全文唯一通道走 rawRequest）。
func slimInputSnapshot(snapshot json.RawMessage) json.RawMessage {
	if len(snapshot) == 0 {
		return nil
	}
	var parsed map[string]json.RawMessage
	if json.Unmarshal(snapshot, &parsed) != nil {
		return nil
	}
	slim := make(map[string]json.RawMessage, len(parsed))
	if messages, ok := parsed["messages"]; ok {
		var fields map[string]json.RawMessage
		if json.Unmarshal(messages, &fields) == nil {
			delete(fields, "items")
			if payload, err := json.Marshal(fields); err == nil {
				slim["messages"] = payload
			}
		}
	}
	for _, key := range []string{"system_tokens", "tool_count", "user_id"} {
		if value, ok := parsed[key]; ok {
			slim[key] = value
		}
	}
	payload, err := json.Marshal(slim)
	if err != nil {
		return nil
	}
	return payload
}

// ==================== 资源消耗 / 性能趋势 ====================

// Costs 资源消耗聚合：按模型/智能体/用户维度分页聚合 + 按日 Token 趋势。
func (s *ObservabilityService) Costs(ctx context.Context, q *bo.CostsQuery) (*vo.ObservabilityCostsVO, error) {
	page, size := q.PageNum, q.PageSize
	dimension := q.Dimension
	if dimension == "" {
		dimension = "model"
	}
	if dimension != "model" && dimension != "agent" && dimension != "user" {
		return nil, common.NewBizError(common.PARAM_ERROR, "dimension 仅支持 model/agent/user")
	}
	start, end, err := observabilityRange(q.StartTime, q.EndTime)
	if err != nil {
		return nil, err
	}

	rows, total, err := s.repo.PaginateCostAgg(ctx, dimension, start, end, page, size)
	if err != nil {
		return nil, err
	}
	items := make([]vo.CostItemVO, 0, len(rows))
	for i := range rows {
		row := &rows[i]
		item := vo.CostItemVO{
			TraceCount: row.TraceCount, TotalTokens: row.TotalTokens,
			PromptTokens: row.PromptTokens, CompletionTokens: row.CompletionTokens,
			CachedTokens: row.CachedTokens,
		}
		switch dimension {
		case "model":
			item.Model = row.Dimension
		case "agent":
			item.AgentCode = row.Dimension
		case "user":
			if row.Dimension != nil {
				if userID, convErr := strconv.ParseInt(*row.Dimension, 10, 64); convErr == nil {
					item.UserID = &userID
				}
			}
		}
		items = append(items, item)
	}

	trendRows, err := s.repo.CostTrendByDay(ctx, start, end)
	if err != nil {
		return nil, err
	}
	trend := make([]vo.CostTrendItemVO, 0, len(trendRows))
	for _, row := range trendRows {
		trend = append(trend, vo.CostTrendItemVO{
			Date: row.Date, TraceCount: row.TraceCount, TotalTokens: row.TotalTokens,
			PromptTokens: row.PromptTokens, CompletionTokens: row.CompletionTokens,
			CachedTokens: row.CachedTokens,
		})
	}
	return &vo.ObservabilityCostsVO{Items: items, Total: total, Trend: trend}, nil
}

// Trends 性能趋势：按维度+日期聚合调用量/成功率/平均延迟（首 Token 延迟取成功调用口径）。
func (s *ObservabilityService) Trends(ctx context.Context, q *bo.TrendsQuery) ([]vo.TrendVO, error) {
	dimension := q.Dimension
	if dimension == "" {
		dimension = "model"
	}
	if dimension != "model" && dimension != "agent" {
		return nil, common.NewBizError(common.PARAM_ERROR, "dimension 仅支持 model/agent")
	}
	start, end, err := observabilityRange(q.StartTime, q.EndTime)
	if err != nil {
		return nil, err
	}
	rows, err := s.repo.PerformanceTrends(ctx, dimension, start, end)
	if err != nil {
		return nil, err
	}
	items := make([]vo.TrendVO, 0, len(rows))
	for i := range rows {
		row := &rows[i]
		item := vo.TrendVO{
			Date: row.Date, CallCount: row.CallCount, SuccessCount: row.SuccessCount,
		}
		if row.CallCount > 0 {
			item.SuccessRate = round2(float64(row.SuccessCount) / float64(row.CallCount) * 100)
		}
		if row.AvgFirstTokenMs != nil {
			value := round2(*row.AvgFirstTokenMs)
			item.AvgFirstTokenMs = &value
		}
		if row.AvgDurationMs != nil {
			value := round2(*row.AvgDurationMs)
			item.AvgDurationMs = &value
		}
		if dimension == "model" {
			item.Model = row.Dimension
		} else {
			item.AgentCode = row.Dimension
		}
		items = append(items, item)
	}
	return items, nil
}

// ==================== 导出 ====================

// ExportTraces 过程链导出（CSV，UTF-8 BOM 便于 Excel 打开），按检索条件全量导出并限行数。
func (s *ObservabilityService) ExportTraces(ctx context.Context, q *bo.TracePageQuery) ([]byte, error) {
	start, end, err := observabilityRange(q.StartTime, q.EndTime)
	if err != nil {
		return nil, err
	}
	if err := validateTraceQuery(q); err != nil {
		return nil, err
	}
	filter := airepo.FilterFromQuery(q, start, end)
	count, err := s.repo.CountTraces(ctx, filter)
	if err != nil {
		return nil, err
	}
	if count > observabilityMaxRows {
		return nil, common.NewBizError(
			common.EXPORT_ROWS_EXCEED_LIMIT,
			fmt.Sprintf("导出行数 %d 超出限制 %d", count, observabilityMaxRows),
		)
	}
	traces, err := s.repo.ListTracesForExport(ctx, filter)
	if err != nil {
		return nil, err
	}

	var buf bytes.Buffer
	buf.WriteString("\xef\xbb\xbf")
	writer := csv.NewWriter(&buf)
	writer.UseCRLF = true
	_ = writer.Write([]string{
		"trace_id", "conversation_id", "message_id", "agent_code", "model", "status",
		"error_type", "duration_ms", "first_token_ms", "llm_call_count", "total_tokens",
		"prompt_tokens", "completion_tokens", "cached_tokens", "step_count", "create_time",
	})
	for i := range traces {
		trace := &traces[i]
		_ = writer.Write([]string{
			trace.TraceID,
			strconv.FormatInt(trace.ConversationID, 10),
			optionalInt64Text(trace.MessageID),
			optionalText(trace.AgentCode),
			optionalText(trace.Model),
			strconv.Itoa(trace.Status),
			optionalText(trace.ErrorType),
			strconv.Itoa(trace.DurationMs),
			optionalIntText(trace.FirstTokenMs),
			strconv.Itoa(trace.LlmCallCount),
			strconv.Itoa(trace.TotalTokens),
			strconv.Itoa(trace.PromptTokens),
			strconv.Itoa(trace.CompletionTokens),
			strconv.Itoa(trace.CachedTokens),
			strconv.Itoa(trace.StepCount),
			trace.CreateTime.Format("2006-01-02 15:04:05"),
		})
	}
	writer.Flush()
	return buf.Bytes(), nil
}

func optionalText(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func optionalIntText(value *int) string {
	if value == nil {
		return ""
	}
	return strconv.Itoa(*value)
}

func optionalInt64Text(value *int64) string {
	if value == nil {
		return ""
	}
	return strconv.FormatInt(*value, 10)
}

// ==================== 辅助 ====================

func traceToVO(t *model.SysAiTrace) vo.TraceVO {
	return vo.TraceVO{
		TraceID: t.TraceID, ConversationID: t.ConversationID, MessageID: t.MessageID,
		AgentCode: t.AgentCode, TraceType: t.TraceType, Model: t.Model, Status: t.Status,
		ErrorType: t.ErrorType, DurationMs: t.DurationMs, FirstTokenMs: t.FirstTokenMs,
		LlmCallCount: t.LlmCallCount, TotalTokens: t.TotalTokens, PromptTokens: t.PromptTokens,
		CompletionTokens: t.CompletionTokens, CachedTokens: t.CachedTokens,
		StepCount: t.StepCount, CreateTime: t.CreateTime,
	}
}

func llmCallToVO(c *model.SysAiLlmCall) vo.LlmCallVO {
	return vo.LlmCallVO{
		Seq: c.Seq, StepPosition: c.StepPosition, Model: c.Model, Status: c.Status,
		ErrorType: c.ErrorType, DurationMs: c.DurationMs, FirstTokenMs: c.FirstTokenMs,
		PromptTokens: c.PromptTokens, CompletionTokens: c.CompletionTokens,
		CachedTokens: c.CachedTokens, ToolCall: c.ToolCall, InputSnapshot: c.InputSnapshot,
		OutputSnapshot: c.OutputSnapshot, Attempts: c.Attempts, StartTime: c.StartTime,
		RawRequest: c.RawRequest, RawResponse: c.RawResponse, CreateTime: c.CreateTime,
	}
}

func thoughtToVO(t *model.SysAiAgentThought) vo.AgentThoughtVO {
	latency := 0
	if t.LatencyMs != nil {
		latency = *t.LatencyMs
	}
	return vo.AgentThoughtVO{
		ID: t.ID, MessageID: t.MessageID, ConversationID: t.ConversationID,
		Position: t.Position, AgentCode: t.AgentCode, IsSubagent: t.IsSubagent,
		Thought: t.Thought, Tool: t.Tool, ToolInput: json.RawMessage(t.ToolInput),
		Observation: t.Observation, Status: t.Status, LatencyMs: latency,
		Error: t.Error, CreateTime: t.CreateTime,
	}
}

func billingToVO(b *model.SysAiBilling) vo.TraceBillingVO {
	return vo.TraceBillingVO{
		BillType: &b.BillType, Model: &b.Model, ActualModel: b.ActualModel,
		ProviderID: b.ProviderID, InputTokens: b.InputTokens, OutputTokens: b.OutputTokens,
		CachedInputTokens: b.CachedInputTokens, Credits: b.Credits, CreditsSaved: b.CreditsSaved,
		ErrorCode: b.ErrorCode, LatencyMs: b.LatencyMs, RequestID: b.RequestID,
		CreateTime: b.CreateTime,
	}
}

func artifactToVO(a *model.SysAiArtifact) vo.TraceArtifactVO {
	return vo.TraceArtifactVO{
		ID: a.ID, Type: &a.Type, Summary: json.RawMessage(a.Summary),
		RefType: a.RefType, RefID: a.RefID, CreateTime: a.CreateTime,
	}
}

func timelineMessageToVO(m *model.SysAiMessage) *vo.TimelineMessageVO {
	if m == nil {
		return nil
	}
	created := m.CreateTime
	return &vo.TimelineMessageVO{
		ID: m.ID, Role: m.Role, Content: m.Content, Status: m.Status, Model: m.Model,
		InputTokens: m.InputTokens, OutputTokens: m.OutputTokens, CreateTime: &created,
	}
}

func traceMessageToVO(m *model.SysAiMessage) vo.TraceMessageVO {
	return vo.TraceMessageVO{
		ID: m.ID, ConversationID: m.ConversationID, ParentMessageID: m.ParentMessageID,
		Role: m.Role, Content: m.Content, Status: m.Status, Model: m.Model,
		InputTokens: m.InputTokens, OutputTokens: m.OutputTokens, CreateTime: m.CreateTime,
	}
}

// validateTraceQuery 过程链检索的枚举参数校验（对齐 python TracePageQuery 的 ge/le 与 Literal 约束）。
func validateTraceQuery(q *bo.TracePageQuery) error {
	if q.Status != nil && (*q.Status < 1 || *q.Status > 4) {
		return common.NewBizError(common.PARAM_ERROR, "status 仅支持 1/2/3/4")
	}
	if q.Capability != "" && q.Capability != "memory" && q.Capability != "kb" && q.Capability != "tools" {
		return common.NewBizError(common.PARAM_ERROR, "capability 仅支持 memory/kb/tools")
	}
	return nil
}

// observabilityRange 解析时间筛选区间（非法格式报参数错误）。
func observabilityRange(startText, endText string) (*time.Time, *time.Time, error) {
	start, err := parseObservabilityTime(startText)
	if err != nil {
		return nil, nil, err
	}
	end, err := parseObservabilityTime(endText)
	if err != nil {
		return nil, nil, err
	}
	return start, end, nil
}

// parseObservabilityTime 解析前端传入的日期时间字符串，非法格式返回参数错误。
func parseObservabilityTime(value string) (*time.Time, error) {
	if value == "" {
		return nil, nil
	}
	for _, layout := range []string{
		time.RFC3339, "2006-01-02T15:04:05", "2006-01-02 15:04:05", "2006-01-02",
	} {
		if parsed, err := time.ParseInLocation(layout, value, time.Local); err == nil {
			return &parsed, nil
		}
	}
	return nil, common.NewBizError(common.PARAM_ERROR, "时间格式不正确: "+value)
}

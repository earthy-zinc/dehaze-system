package ai

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	aidomainrepo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// ==================== 纯逻辑：轮次切分与过程链归属 ====================

func TestSplitTimelineRounds(t *testing.T) {
	chain := []model.SysAiMessage{
		{ID: 1, Role: "assistant"},
		{ID: 2, Role: "user"},
		{ID: 3, Role: "assistant"},
		{ID: 4, Role: "user"},
		{ID: 5, Role: "assistant"},
	}
	rounds := splitTimelineRounds(chain)
	require.Len(t, rounds, 3, "链首 assistant 单独成轮，其后每个 user 开一轮")
	require.Nil(t, rounds[0].user)
	require.EqualValues(t, 1, rounds[0].assistant.ID)
	require.EqualValues(t, 2, rounds[1].user.ID)
	require.EqualValues(t, 3, rounds[1].assistant.ID)
	require.EqualValues(t, 4, rounds[2].user.ID)
	require.EqualValues(t, 5, rounds[2].assistant.ID)
}

func TestAttachTracesToRounds(t *testing.T) {
	base := time.Date(2026, 9, 17, 10, 0, 0, 0, time.Local)
	chain := []model.SysAiMessage{
		{ID: 1, Role: "user", CreateTime: base},
		{ID: 2, Role: "assistant", CreateTime: base.Add(time.Second)},
		{ID: 3, Role: "user", CreateTime: base.Add(2 * time.Second)},
		{ID: 4, Role: "assistant", CreateTime: base.Add(3 * time.Second)},
	}
	rounds := splitTimelineRounds(chain)

	messageID := int64(2)
	traces := []model.SysAiTrace{
		// 无 message_id 的旁路：按 create_time 挂到触发时点之前（含）的最近轮次（第一轮窗口内）
		{TraceID: "sidecar", TraceType: "memory_extraction", CreateTime: base.Add(1500 * time.Millisecond)},
		// 命中第一轮 assistant 消息
		{TraceID: "main", TraceType: "conversation", MessageID: &messageID, CreateTime: base.Add(time.Second)},
	}
	attachTracesToRounds(rounds, traces)

	require.Len(t, rounds[0].traces, 2)
	require.Equal(t, "main", rounds[0].traces[0].TraceID, "主对话 trace 必须排在旁路之前")
	require.Equal(t, "sidecar", rounds[0].traces[1].TraceID)
	require.Empty(t, rounds[1].traces, "旁路只挂触发时点之前的最近轮次，不落入后续轮次")
}

func TestAttachTracesFallsBackToFirstRound(t *testing.T) {
	base := time.Date(2026, 9, 17, 10, 0, 0, 0, time.Local)
	chain := []model.SysAiMessage{
		{ID: 1, Role: "user", CreateTime: base},
		{ID: 2, Role: "assistant", CreateTime: base.Add(time.Second)},
	}
	rounds := splitTimelineRounds(chain)
	// 触发时点早于全部轮次 → 落到首轮
	attachTracesToRounds(rounds, []model.SysAiTrace{
		{TraceID: "orphan", TraceType: "summary", CreateTime: base.Add(-time.Second)},
	})
	require.Len(t, rounds[0].traces, 1)
	require.Equal(t, "orphan", rounds[0].traces[0].TraceID)
}

// ==================== 纯逻辑：事件交织与快照瘦身 ====================

func TestTraceEventsOrderingAndRawSwitch(t *testing.T) {
	svc := &ObservabilityService{}
	created := time.Date(2026, 9, 17, 10, 0, 0, 0, time.Local)
	// duration=2000ms → context 近似起点与用户输入同刻，验证同刻按业务序
	trace := &model.SysAiTrace{
		TraceID: "t1", ConversationID: 1, TraceType: "conversation", Model: strPtr("m1"),
		Status: 1, DurationMs: 2000, CreateTime: created,
		ContextSnapshot: json.RawMessage(
			`{"items":[{"type":"memory","tokens":12}],"events":[{"event":"guardrail","rule":"pii_mask"}]}`),
	}
	start := created.Add(-1500 * time.Millisecond)
	seq := 1
	calls := []model.SysAiLlmCall{
		{
			TraceID: "t1", Seq: seq, StartTime: &start, Model: strPtr("m1"), Status: 1,
			DurationMs: 1200, PromptTokens: 10, CompletionTokens: 5, CachedTokens: 2,
			InputSnapshot: json.RawMessage(`{"messages":{"items":[{"role":"user","content":"你好"}],` +
				`"counts":{"user":1},"tokens":10},"system_content":"系统","system_tokens":20,` +
				`"tool_count":1,"tools":[{"name":"web_search"}],"user_id":1}`),
			OutputSnapshot: json.RawMessage(`{"text":"回复"}`),
			RawRequest:     json.RawMessage(`{"model":"m1"}`),
			RawResponse:    json.RawMessage(`{"choices":[]}`),
		},
		// start_time 为 NULL 的调用：ts 缺失必须沉底
		{TraceID: "t1", Seq: 2, Status: 1, DurationMs: 300, PromptTokens: 1},
	}
	userMessage := &model.SysAiMessage{ID: 7, Role: "user", CreateTime: created.Add(-2 * time.Second)}

	events := svc.traceEvents(trace, calls, nil, nil, userMessage, false)
	kinds := make([]string, 0, len(events))
	for i := range events {
		kinds = append(kinds, events[i].Kind)
	}
	require.Equal(t,
		[]string{"input", "context", "system_event", "llm_call", "llm_call"}, kinds,
		"同刻事件按 input→context→system_event→llm_call 业务序，ts 缺失的调用沉底")

	input := events[0]
	require.EqualValues(t, 7, input.Message.ID)
	contextEvent := events[1]
	require.NotEmpty(t, contextEvent.Snapshot)
	systemEvent := events[2]
	require.Equal(t, "guardrail", *systemEvent.Event)
	require.NotEmpty(t, systemEvent.Detail)

	first := events[3]
	require.Equal(t, 1, *first.Seq)
	require.Nil(t, first.RawRequest, "include=false 时不返回原始报文")
	require.Nil(t, first.RawResponse)
	// summary 为纯计数单形状：不含 messages.items 全文与 system_content/tools
	summary := string(first.Summary)
	require.NotContains(t, summary, "你好")
	require.NotContains(t, summary, "系统")
	require.NotContains(t, summary, "web_search")
	require.Contains(t, summary, `"system_tokens":20`)
	require.Contains(t, summary, `"tool_count":1`)
	require.Contains(t, summary, `"user_id":1`)
	require.Contains(t, summary, `"tokens":10`)
	require.Contains(t, summary, `"outputSnapshot"`)

	withRaw := svc.traceEvents(trace, calls, nil, nil, userMessage, true)
	require.NotEmpty(t, withRaw[3].RawRequest, "include=true 时返回 wire 级原始请求体")
	require.NotEmpty(t, withRaw[3].RawResponse)
}

func TestTraceEventsIncludesBillingAndThought(t *testing.T) {
	svc := &ObservabilityService{}
	created := time.Date(2026, 9, 17, 10, 0, 0, 0, time.Local)
	trace := &model.SysAiTrace{TraceID: "t2", ConversationID: 1, Status: 1, CreateTime: created}
	latency := 120
	thoughts := []model.SysAiAgentThought{{
		ID: 1, MessageID: 2, ConversationID: 1, Position: 1, Tool: strPtr("web_search"),
		ToolInput: `{"q":"dehaze"}`, Status: 1, LatencyMs: &latency, CreateTime: created.Add(200 * time.Millisecond),
	}}
	billing := []model.SysAiBilling{{
		ID: 1, UserID: 1, BillType: "chat", Model: "m1", InputTokens: 10, OutputTokens: 5,
		CachedInputTokens: 2, Credits: 7, CreateTime: created.Add(400 * time.Millisecond),
	}}
	events := svc.traceEvents(trace, nil, thoughts, billing, nil, false)
	require.Len(t, events, 3)
	require.Equal(t, "tool_exec", events[1].Kind)
	require.Equal(t, 1, *events[1].Position)
	require.Equal(t, 120, *events[1].LatencyMs)
	require.Equal(t, "billing", events[2].Kind)
	require.Equal(t, 7, *events[2].Credits)
	require.JSONEq(t, `{"input":10,"output":5,"cached":2}`, string(events[2].Tokens))
}

func TestBuildCallSummaryEmpty(t *testing.T) {
	require.Nil(t, buildCallSummary(nil, nil), "无输入与输出快照时不产出摘要对象")
	require.Nil(t, slimInputSnapshot(json.RawMessage(`not-json`)))
}

// ==================== 纯逻辑：查询参数校验 ====================

// 分页参数的缺省与越界拦截由 handler 的 parsePaginationWithSize 统一负责（见
// internal/api/ai_observability_sweep_test.go），service 只按透传值分页，故此处不再重复校验。

func TestParseObservabilityTime(t *testing.T) {
	parsed, err := parseObservabilityTime("2026-09-17 10:20:30")
	require.NoError(t, err)
	require.Equal(t, 10, parsed.Hour())

	parsed, err = parseObservabilityTime("2026-09-17")
	require.NoError(t, err)
	require.Equal(t, 17, parsed.Day())

	parsed, err = parseObservabilityTime("2026-09-17T10:20:30+08:00")
	require.NoError(t, err)
	require.Equal(t, 2026, parsed.Year())

	parsed, err = parseObservabilityTime("")
	require.NoError(t, err)
	require.Nil(t, parsed)

	_, err = parseObservabilityTime("2026/09/17")
	requireBizCode(t, err, common.PARAM_ERROR)
}

func TestValidateTraceQuery(t *testing.T) {
	require.NoError(t, validateTraceQuery(&bo.TracePageQuery{}))
	require.NoError(t, validateTraceQuery(&bo.TracePageQuery{Status: intPtr(4), Capability: "kb"}))
	requireBizCode(t, validateTraceQuery(&bo.TracePageQuery{Status: intPtr(5)}), common.PARAM_ERROR)
	requireBizCode(t, validateTraceQuery(&bo.TracePageQuery{Capability: "vector"}), common.PARAM_ERROR)
}

// ==================== 仓储 + 服务：真实 MySQL ====================

type observabilityFixture struct {
	conversationID int64
	assistantID    int64
	traceID        string
	sidecarTraceID string
	quotaTraceID   string
	model          string
	ownerID        int64
}

func TestObservabilityServiceRealDB(t *testing.T) {
	db := testutil.NewTestDB(t)
	ctx := context.Background()
	svc := newObservabilityServiceForTest(db)
	before, err := svc.Summary(ctx)
	require.NoError(t, err)

	fixture := seedObservabilityFixture(t, db)

	// ---------- 总览 ----------
	after, err := svc.Summary(ctx)
	require.NoError(t, err)
	require.Equal(t, before.Total+3, after.Total, "主过程链 + 旁路 + 配额拒绝过程链")
	require.Equal(t, before.SuccessCount+2, after.SuccessCount)
	require.Equal(t, before.QuotaRejected+1, after.QuotaRejected, "error_type=quota 计入配额拒绝")
	require.Equal(t, before.HighRiskCalls+1, after.HighRiskCalls, "step_count 超阈值计为高风险")

	// ---------- 过程链检索 ----------
	// 分页缺省值（python BasePageQuery 1/10）由 handler 的 parsePaginationWithSize 解析，service 只按透传值分页
	page, err := svc.ListTraces(ctx, &bo.TracePageQuery{
		PageNum: 1, PageSize: 10, ConversationID: &fixture.conversationID,
	})
	require.NoError(t, err)
	require.EqualValues(t, 3, page.Total)
	require.Len(t, page.List, 3)
	require.NotNil(t, page.List[0].ConversationTitle, "会话标题必须回填")
	require.Contains(t, *page.List[0].ConversationTitle, "可观测性测试会话")

	page, err = svc.ListTraces(ctx, &bo.TracePageQuery{
		PageNum: 1, PageSize: 10, ConversationID: &fixture.conversationID, Keyword: fixture.traceID,
	})
	require.NoError(t, err)
	require.EqualValues(t, 1, page.Total, "关键词命中 trace_id")

	page, err = svc.ListTraces(ctx, &bo.TracePageQuery{
		PageNum: 1, PageSize: 10, ConversationID: &fixture.conversationID, Status: intPtr(4),
	})
	require.NoError(t, err)
	require.EqualValues(t, 0, page.Total, "无超时过程链")

	page, err = svc.ListTraces(ctx, &bo.TracePageQuery{
		PageNum: 1, PageSize: 10, ConversationID: &fixture.conversationID, Status: intPtr(2),
	})
	require.NoError(t, err)
	require.EqualValues(t, 1, page.Total, "配额拒绝过程链为失败状态")

	page, err = svc.ListTraces(ctx, &bo.TracePageQuery{
		PageNum: 1, PageSize: 10, ConversationID: &fixture.conversationID, Capability: "memory",
	})
	require.NoError(t, err)
	require.EqualValues(t, 1, page.Total, "能力维度匹配 context_snapshot.items[].type")

	page, err = svc.ListTraces(ctx, &bo.TracePageQuery{
		PageNum: 1, PageSize: 10, ConversationID: &fixture.conversationID, Status: intPtr(5),
	})
	requireBizCode(t, err, common.PARAM_ERROR)

	// ---------- 过程链详情 ----------
	detail, err := svc.GetTrace(ctx, fixture.traceID, fixture.ownerID, true)
	require.NoError(t, err)
	require.Equal(t, fixture.traceID, detail.TraceID)
	require.Equal(t, "conversation", detail.TraceType)
	require.NotEmpty(t, detail.ContextSnapshot)
	require.Len(t, detail.LlmCalls, 1)
	require.Equal(t, 1, detail.LlmCalls[0].Seq)
	require.Len(t, detail.Thoughts, 1)
	require.Equal(t, "web_search", *detail.Thoughts[0].Tool)
	require.Len(t, detail.Billing, 1, "计费按 request_id=trace_id 精确归因")
	require.Equal(t, "chat", *detail.Billing[0].BillType)
	require.Len(t, detail.Artifacts, 1)
	require.Equal(t, "metric_report", *detail.Artifacts[0].Type)
	require.Len(t, detail.Messages, 2, "详情回放所属会话消息（时间正序）")
	require.Equal(t, "user", detail.Messages[0].Role)

	_, err = svc.GetTrace(ctx, fixture.traceID, fixture.ownerID, false)
	require.NoError(t, err, "普通用户可查自己会话的过程链")

	_, err = svc.GetTrace(ctx, fixture.traceID, fixture.ownerID+999999, false)
	requireBizCode(t, err, common.RESOURCE_NOT_FOUND)

	_, err = svc.GetTrace(ctx, "not-exist-trace", fixture.ownerID, true)
	requireBizCode(t, err, common.RESOURCE_NOT_FOUND)

	// ---------- 会话审计时间线 ----------
	timeline, err := svc.ConversationTimeline(ctx, fixture.conversationID, fixture.ownerID, true, true)
	require.NoError(t, err)
	require.Equal(t, fixture.conversationID, timeline.Conversation.ID)
	require.Len(t, timeline.Rounds, 1)
	require.NotNil(t, timeline.Rounds[0].UserMessage)
	require.NotNil(t, timeline.Rounds[0].AssistantMessage)
	require.Len(t, timeline.Rounds[0].Traces, 3, "轮次内主对话 + 两条旁路")
	require.Equal(t, "conversation", timeline.Rounds[0].Traces[0].TraceType, "主对话 trace 排最前")
	require.Equal(t, "memory_extraction", timeline.Rounds[0].Traces[1].TraceType)
	require.Equal(t, "summary", timeline.Rounds[0].Traces[2].TraceType)

	eventKinds := map[string]bool{}
	for _, event := range timeline.Rounds[0].Traces[0].Events {
		eventKinds[event.Kind] = true
	}
	for _, kind := range []string{"input", "context", "system_event", "llm_call", "tool_exec", "billing"} {
		require.True(t, eventKinds[kind], "时间线缺少事件类型 %s", kind)
	}

	light, err := svc.ConversationTimeline(ctx, fixture.conversationID, fixture.ownerID, true, false)
	require.NoError(t, err)
	require.Nil(t, light.Rounds[0].Traces[0].Events[3].RawRequest, "include=false 不返回原始报文")

	_, err = svc.ConversationTimeline(ctx, fixture.conversationID, fixture.ownerID+999999, false, true)
	requireBizCode(t, err, common.RESOURCE_NOT_FOUND)

	// ---------- 资源消耗 / 性能趋势 ----------
	costs, err := svc.Costs(ctx, &bo.CostsQuery{PageNum: 1, PageSize: 10, Dimension: "model"})
	require.NoError(t, err)
	require.GreaterOrEqual(t, costs.Total, int64(1))
	modelCost := findCostItem(costs.Items, fixture.model)
	require.NotNil(t, modelCost, "按模型聚合应包含测试模型")
	require.GreaterOrEqual(t, modelCost.TraceCount, int64(3))
	require.GreaterOrEqual(t, modelCost.TotalTokens, int64(15))
	require.NotEmpty(t, costs.Trend, "按日 Token 消耗趋势非空")

	costs, err = svc.Costs(ctx, &bo.CostsQuery{PageNum: 1, PageSize: 10, Dimension: "user"})
	require.NoError(t, err)
	require.GreaterOrEqual(t, costs.Total, int64(1))
	require.NotNil(t, findCostItem(costs.Items, ""))

	costs, err = svc.Costs(ctx, &bo.CostsQuery{PageNum: 1, PageSize: 10, Dimension: "agent"})
	require.NoError(t, err)
	require.GreaterOrEqual(t, costs.Total, int64(1))

	_, err = svc.Costs(ctx, &bo.CostsQuery{PageNum: 1, PageSize: 10, Dimension: "vector"})
	requireBizCode(t, err, common.PARAM_ERROR)

	trends, err := svc.Trends(ctx, &bo.TrendsQuery{Dimension: "model"})
	require.NoError(t, err)
	require.NotEmpty(t, trends)
	found := false
	for i := range trends {
		if trends[i].Model != nil && *trends[i].Model == fixture.model {
			found = true
			require.GreaterOrEqual(t, trends[i].CallCount, int64(3))
			require.GreaterOrEqual(t, trends[i].SuccessCount, int64(2))
			require.InDelta(t, 66.67, trends[i].SuccessRate, 0.01)
			require.NotNil(t, trends[i].AvgFirstTokenMs, "首 Token 延迟取成功调用口径")
		}
	}
	require.True(t, found, "按模型性能趋势应包含测试模型")

	_, err = svc.Trends(ctx, &bo.TrendsQuery{Dimension: "user"})
	requireBizCode(t, err, common.PARAM_ERROR)

	// ---------- 过程链导出 ----------
	payload, err := svc.ExportTraces(ctx, &bo.TracePageQuery{ConversationID: &fixture.conversationID})
	require.NoError(t, err)
	require.True(t, bytes.HasPrefix(payload, []byte("\xef\xbb\xbf")), "CSV 需带 UTF-8 BOM")
	records, err := csv.NewReader(bytes.NewReader(bytes.TrimPrefix(payload, []byte("\xef\xbb\xbf")))).ReadAll()
	require.NoError(t, err)
	require.Len(t, records, 4, "表头 + 3 条过程链")
	require.Equal(t,
		"trace_id,conversation_id,message_id,agent_code,model,status,error_type,duration_ms,"+
			"first_token_ms,llm_call_count,total_tokens,prompt_tokens,completion_tokens,cached_tokens,"+
			"step_count,create_time", strings.Join(records[0], ","))
	exported := []string{records[1][0], records[2][0], records[3][0]}
	require.ElementsMatch(t, []string{fixture.traceID, fixture.sidecarTraceID, fixture.quotaTraceID}, exported)
	require.Equal(t, fixture.quotaTraceID, records[1][0], "导出与检索同序：create_time/id 倒序")
}

func TestExportTracesInvalidTime(t *testing.T) {
	svc := &ObservabilityService{}
	_, err := svc.ExportTraces(context.Background(), &bo.TracePageQuery{StartTime: "非法时间"})
	requireBizCode(t, err, common.PARAM_ERROR)
}

func newObservabilityServiceForTest(db *gorm.DB) *ObservabilityService {
	return NewObservabilityService(
		airepo.NewObservabilityRepository(db),
		airepo.NewBillingRepository(db),
		aidomainrepo.NewConversationRepository(db),
		aidomainrepo.NewMessageRepository(db),
		aidomainrepo.NewThoughtRepository(db),
	)
}

// seedObservabilityFixture 落一条完整会话（用户/助手消息 + 主过程链 + 旁路 + 配额拒绝链 +
// LLM 调用 + 推理步骤 + 计费 + 产物），覆盖详情/时间线/聚合/导出全链路读取。
func seedObservabilityFixture(t *testing.T, db *gorm.DB) observabilityFixture {
	t.Helper()
	ctx := context.Background()
	suffix := strconv.FormatInt(time.Now().UnixNano(), 10)
	base := time.Now().Add(-time.Minute).Truncate(time.Second)

	var ownerID int64
	require.NoError(t, db.WithContext(ctx).Table("sys_user").
		Select("id").Where("username = ?", "admin").Limit(1).Scan(&ownerID).Error)
	require.NotZero(t, ownerID, "测试库缺少 admin 种子用户")

	conversation := model.SysAiConversation{
		UserID: ownerID, Title: "可观测性测试会话" + suffix, ModelConfig: "{}", CreateTime: base,
	}
	require.NoError(t, db.WithContext(ctx).Create(&conversation).Error)

	userMessage := model.SysAiMessage{
		ConversationID: conversation.ID, Role: "user", Content: strPtr("请分析这张图"),
		ToolCalls: "[]", Metadata: "{}", UsedMemoryIDs: "[]", Status: 2, CreateTime: base,
	}
	require.NoError(t, db.WithContext(ctx).Create(&userMessage).Error)

	assistantMessage := model.SysAiMessage{
		ConversationID: conversation.ID, ParentMessageID: &userMessage.ID, Role: "assistant",
		Content: strPtr("已生成指标报告"), ToolCalls: "[]", Metadata: "{}", UsedMemoryIDs: "[]",
		Model: strPtr("m-observability"), Status: 2, InputTokens: 10, OutputTokens: 5,
		CreateTime: base.Add(time.Second),
	}
	require.NoError(t, db.WithContext(ctx).Create(&assistantMessage).Error)
	require.NoError(t, db.WithContext(ctx).Model(&model.SysAiConversation{}).
		Where("id = ?", conversation.ID).
		Updates(map[string]any{"current_branch_message_id": assistantMessage.ID, "message_count": 2}).Error)

	traceID := "trace-" + suffix
	mainTrace := model.SysAiTrace{
		TraceID: traceID, ConversationID: conversation.ID, MessageID: &assistantMessage.ID,
		AgentCode: strPtr("default"), TraceType: "conversation", Model: strPtr("m-observability"),
		Status: 1, DurationMs: 2000, FirstTokenMs: intPtr(300), LlmCallCount: 1,
		TotalTokens: 15, PromptTokens: 10, CompletionTokens: 5, CachedTokens: 2, StepCount: 3,
		ContextSnapshot: json.RawMessage(
			`{"items":[{"type":"memory","tokens":12},{"type":"history","tokens":30}],` +
				`"events":[{"event":"guardrail","rule":"pii_mask"}]}`),
		CreateTime: base.Add(time.Second),
	}
	require.NoError(t, db.WithContext(ctx).Create(&mainTrace).Error)

	startTime := base.Add(1200 * time.Millisecond)
	call := model.SysAiLlmCall{
		TraceID: traceID, Seq: 1, StepPosition: intPtr(1), Model: strPtr("m-observability"),
		StartTime: &startTime, Status: 1, DurationMs: 1200, FirstTokenMs: intPtr(300),
		PromptTokens: 10, CompletionTokens: 5, CachedTokens: 2,
		InputSnapshot: json.RawMessage(`{"messages":{"items":[{"role":"user","content":"请分析这张图"}],` +
			`"counts":{"user":1},"tokens":10},"system_content":"系统","system_tokens":20,` +
			`"tool_count":1,"tools":[{"name":"web_search"}],"user_id":1}`),
		OutputSnapshot: json.RawMessage(`{"text":"已生成指标报告"}`),
		RawRequest:     json.RawMessage(`{"model":"m-observability"}`),
		RawResponse:    json.RawMessage(`{"choices":[]}`),
		CreateTime:     base.Add(time.Second),
	}
	require.NoError(t, db.WithContext(ctx).Create(&call).Error)

	latency := 120
	thought := model.SysAiAgentThought{
		MessageID: assistantMessage.ID, ConversationID: conversation.ID, Position: 1,
		AgentCode: strPtr("default"), Thought: strPtr("需要检索指标口径"), Tool: strPtr("web_search"),
		ToolInput: `{"q":"dehaze"}`, Observation: strPtr("ok"), Status: 1,
		LatencyMs: &latency, CreateTime: base.Add(1400 * time.Millisecond),
	}
	require.NoError(t, db.WithContext(ctx).Create(&thought).Error)

	billing := model.SysAiBilling{
		UserID: ownerID, ConversationID: &conversation.ID, MessageID: &assistantMessage.ID,
		RequestID: &traceID, Model: "m-observability", BillType: "chat",
		InputTokens: 10, CachedInputTokens: 2, OutputTokens: 5, Credits: 7, CreditsSaved: 1,
		QuotaConsumed: 7, CreateTime: base.Add(1600 * time.Millisecond),
	}
	require.NoError(t, db.WithContext(ctx).Create(&billing).Error)

	artifact := model.SysAiArtifact{
		ConversationID: conversation.ID, MessageID: assistantMessage.ID, Type: "metric_report",
		Summary: `{"psnr":30.5}`, CreateTime: base.Add(1800 * time.Millisecond),
	}
	require.NoError(t, db.WithContext(ctx).Create(&artifact).Error)

	// 旁路过程链（无 message_id，按触发时点挂轮次）
	sidecarID := "trace-sidecar-" + suffix
	require.NoError(t, db.WithContext(ctx).Create(&model.SysAiTrace{
		TraceID: sidecarID, ConversationID: conversation.ID, TraceType: "memory_extraction",
		Model: strPtr("m-observability"), Status: 1, DurationMs: 400,
		CreateTime: base.Add(1500 * time.Millisecond),
	}).Error)

	// 配额拒绝 + 高风险过程链（step_count 超阈值 40）
	quotaID := "trace-quota-" + suffix
	require.NoError(t, db.WithContext(ctx).Create(&model.SysAiTrace{
		TraceID: quotaID, ConversationID: conversation.ID, TraceType: "summary",
		Model: strPtr("m-observability"), Status: 2, ErrorType: strPtr("quota"),
		DurationMs: 100, StepCount: 41,
		CreateTime: base.Add(1900 * time.Millisecond),
	}).Error)

	return observabilityFixture{
		conversationID: conversation.ID,
		assistantID:    assistantMessage.ID,
		traceID:        traceID,
		sidecarTraceID: sidecarID,
		quotaTraceID:   quotaID,
		model:          "m-observability",
		ownerID:        ownerID,
	}
}

func findCostItem(items []vo.CostItemVO, model string) *vo.CostItemVO {
	for i := range items {
		if items[i].Model != nil && *items[i].Model == model {
			return &items[i]
		}
		if model == "" && items[i].UserID != nil {
			return &items[i]
		}
	}
	return nil
}

func requireBizCode(t *testing.T, err error, code *common.ResultCode) {
	t.Helper()
	require.Error(t, err)
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "期望业务错误，实际: %v", err)
	require.Equal(t, code.Code, bizErr.Code().Code)
}

func strPtr(value string) *string { return &value }

func intPtr(value int) *int { return &value }

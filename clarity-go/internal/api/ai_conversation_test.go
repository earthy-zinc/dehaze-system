package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/earthyzinc/clarity-go/internal/model"
	"github.com/earthyzinc/clarity-go/internal/model/aicontract"
	aidomainrepo "github.com/earthyzinc/clarity-go/internal/repository/aidomain"
	aidomain "github.com/earthyzinc/clarity-go/internal/service/aidomain"
	"github.com/earthyzinc/clarity-go/internal/testutil"
	"github.com/earthyzinc/clarity-go/pkg/common"
	"github.com/earthyzinc/clarity-go/pkg/security"
	"github.com/earthyzinc/clarity-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// AI 会话/消息端点的契约回归：消息只有 blocks/plan/pendingInterrupt 一种形状，
// status 为字符串枚举，旧 content/toolCalls/toolCallId/metadata/thoughts 一律不得下发。

// aiConversationFixture 一套可打 HTTP 的会话夹具：四条消息 + 产物。
type aiConversationFixture struct {
	engine      *gin.Engine
	convID      int64
	userMsgID   int64
	assistantID int64
	expiredID   int64
	staleID     int64
	artifactIDs []int64
}

// staticModelResolver 会话服务在 Create/Update 时才解析模型，查询类用例给固定值即可。
type staticModelResolver struct{}

func (staticModelResolver) ResolveChatModel(
	context.Context, int64, *string, string,
) (string, error) {
	return "m-contract", nil
}

func textBlock(id string, seq int64, text string) aicontract.Block {
	return aicontract.Block{ID: id, Kind: aicontract.BlockKindText, Seq: seq, Text: &text}
}

func newAiConversationFixture(t *testing.T) *aiConversationFixture {
	t.Helper()
	db := testutil.NewTestDB(t)
	ctx := context.Background()

	convRepo := aidomainrepo.NewConversationRepository(db)
	msgRepo := aidomainrepo.NewMessageRepository(db)
	artifactRepo := aidomainrepo.NewArtifactRepository(db)
	conversationAPI := NewAiConversationApi(
		aidomain.NewConversationService(convRepo, msgRepo, aidomainrepo.NewAgentRepository(db), staticModelResolver{}),
		aidomain.NewMessageService(convRepo, msgRepo, artifactRepo),
		aidomain.NewFeedbackService(convRepo, msgRepo, aidomainrepo.NewMessageFeedbackRepository(db), aidomainrepo.NewMemoryRepository(db)),
		aidomain.NewArtifactService(artifactRepo, convRepo, msgRepo, nil),
	)

	engine := gin.New()
	engine.Use(func(c *gin.Context) {
		c.Set("claims", &security.CustomClaims{UserID: 910001})
		c.Next()
	})
	engine.GET("/ai/conversations/:id/messages", conversationAPI.ListMessages)
	engine.GET("/ai/conversations/:id/export", conversationAPI.ExportConversation)
	engine.GET("/ai/messages/:id", conversationAPI.GetMessage)
	engine.GET("/ai/messages/:id/interrupt", conversationAPI.GetMessageInterrupt)

	conv := &model.SysAiConversation{
		UserID: 910001, Title: "契约回归会话", TitleSource: "auto", Status: 1, MessageCount: 4,
	}
	require.NoError(t, convRepo.Create(ctx, conv))

	userMessage := &model.SysAiMessage{
		ConversationID: conv.ID, Role: aicontract.RoleUser, Status: aicontract.MessageStatusCompleted,
		Blocks: aicontract.EncodeBlocks([]aicontract.Block{textBlock("u-text", 1, "请分析这张图")}),
	}
	require.NoError(t, msgRepo.Create(ctx, userMessage))

	actor, at := "assistant", "2026-10-21T10:00:00+08:00"
	taskStatus := aicontract.PlanStatusCompleted
	planJSON, err := json.Marshal(aicontract.Plan{
		Phase:  strPtr("collect"),
		Status: aicontract.PlanStatusRunning,
		Tasks: []aicontract.PlanTask{{
			ID: "t1", Description: strPtr("读取直方图"), Status: &taskStatus,
		}},
		Revisions: []aicontract.PlanRevision{{
			RevisionNo: 1, Reason: "初始计划", Actor: &actor, At: &at,
		}},
		AwaitingApproval: true,
	})
	require.NoError(t, err)

	confirmKind, action, reason := aicontract.ConfirmKindAlgorithmRecommend, "run", "需要确认算法"
	interruptJSON, err := json.Marshal(aicontract.Interrupt{
		ID: "it-1", Type: aicontract.InterruptTypeConfirm,
		ConfirmKind: &confirmKind, Action: &action, Reason: &reason,
		Recommendation: &aicontract.Recommendation{ModelID: "alg-1", ModelName: "去雾算法"},
		CreatedAt:      "2026-10-21T10:00:00+08:00",
		ExpiresAt:      time.Now().Add(10 * time.Minute).Format(time.RFC3339),
		Actionable:     true,
	})
	require.NoError(t, err)

	toolInput := map[string]any{"query": "dehaze"}
	assistant := &model.SysAiMessage{
		ConversationID: conv.ID, ParentMessageID: &userMessage.ID,
		Role: aicontract.RoleAssistant, Status: aicontract.MessageStatusAwaitingInput,
		Blocks: aicontract.EncodeBlocks([]aicontract.Block{
			textBlock("a-text", 2, "已生成指标报告"),
			{ID: "a-think", Kind: aicontract.BlockKindThinking, Seq: 1, Text: strPtr("先读直方图")},
			{ID: "a-tool", Kind: aicontract.BlockKindToolUse, Seq: 3, Tool: &aicontract.ToolCall{
				CallID: "call-1", Name: "image_dehaze", Input: toolInput,
				Status: aicontract.ToolStatusSuccess, SubAgent: false,
			}},
		}),
		Plan:             string(planJSON),
		PendingInterrupt: string(interruptJSON),
		Suggestions:      `["继续分析","导出报告"]`,
		Model:            strPtr("m-contract"),
		UsedMemoryIDs:    `[11,12]`,
		InputTokens:      100, OutputTokens: 40, CachedInputTokens: 20, Credits: 7,
		Edited: 1,
	}
	require.NoError(t, msgRepo.Create(ctx, assistant))

	expired := &model.SysAiMessage{
		ConversationID: conv.ID, ParentMessageID: &assistant.ID,
		Role: aicontract.RoleAssistant, Status: aicontract.MessageStatusAwaitingInput,
		PendingInterrupt: `{"id":"it-expired","type":"quota","createdAt":"` +
			time.Now().Add(-time.Hour).Format(time.RFC3339) + `","expiresAt":"` +
			time.Now().Add(-time.Minute).Format(time.RFC3339) + `","actionable":true}`,
	}
	require.NoError(t, msgRepo.Create(ctx, expired))

	// 非 awaiting_input 的助手消息仍残留中断 payload：下发给前端时必须 actionable=false。
	stale := &model.SysAiMessage{
		ConversationID: conv.ID, ParentMessageID: &expired.ID,
		Role: aicontract.RoleAssistant, Status: aicontract.MessageStatusCompleted,
		PendingInterrupt: `{"id":"it-stale","type":"confirm","createdAt":"` +
			time.Now().Add(-time.Minute).Format(time.RFC3339) + `","expiresAt":"` +
			time.Now().Add(time.Hour).Format(time.RFC3339) + `","actionable":true}`,
	}
	require.NoError(t, msgRepo.Create(ctx, stale))

	artifactIDs := make([]int64, 0, 2)
	for _, artifactType := range []string{"metric_report", "image_result"} {
		artifact := &model.SysAiArtifact{
			ConversationID: conv.ID, MessageID: assistant.ID,
			Type: artifactType, Summary: `{"k":1}`,
		}
		require.NoError(t, db.Create(artifact).Error)
		artifactIDs = append(artifactIDs, artifact.ID)
	}
	require.NoError(t, convRepo.UpdateCurrentBranch(ctx, conv.ID, stale.ID, 910001))

	return &aiConversationFixture{
		engine: engine, convID: conv.ID,
		userMsgID: userMessage.ID, assistantID: assistant.ID, expiredID: expired.ID,
		staleID: stale.ID, artifactIDs: artifactIDs,
	}
}

func (f *aiConversationFixture) do(t *testing.T, path string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	f.engine.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
	require.Equal(t, http.StatusOK, rec.Code)
	return rec
}

// decodeData 解出统一响应体里的 data 原始 JSON。
func decodeData(t *testing.T, rec *httptest.ResponseRecorder) json.RawMessage {
	t.Helper()
	var body struct {
		Code string          `json:"code"`
		Data json.RawMessage `json:"data"`
		Msg  string          `json:"msg"`
	}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body), "响应体不是合法 JSON: %s", rec.Body.String())
	require.Equal(t, common.SUCCESS.Code, body.Code, "接口应返回成功: %s", rec.Body.String())
	return body.Data
}

// TestMessageListUsesContractShape 列表按契约下发：blocks/plan/pendingInterrupt/usage 齐备，
// 旧字段 content/toolCalls/toolCallId/metadata/thoughts 一律不出现，status 为字符串。
func TestMessageListUsesContractShape(t *testing.T) {
	f := newAiConversationFixture(t)
	data := decodeData(t, f.do(t, "/ai/conversations/"+pathID(f.convID)+"/messages"))

	raw := string(data)
	for _, legacy := range []string{`"content"`, `"toolCalls"`, `"toolCallId"`, `"metadata"`, `"thoughts"`} {
		require.NotContains(t, raw, legacy, "旧字段 %s 不得出现在消息契约里", legacy)
	}

	var list aicontract.MessageList
	require.NoError(t, json.Unmarshal(data, &list))
	require.EqualValues(t, 4, list.Total)
	require.False(t, list.HasMore)
	require.Len(t, list.List, 4, "四条消息按 id 倒序")
	require.Equal(t, f.staleID, list.List[0].ID, "非 awaiting_input 的残留中断排在最新")
	require.Equal(t, f.expiredID, list.List[1].ID)
	require.Equal(t, f.assistantID, list.List[2].ID)
	require.Equal(t, f.userMsgID, list.List[3].ID, "当前分支消息与其父消息按时间倒序排列")

	// 残留中断（消息已 completed）：payload 保留但 actionable=false
	stale := list.List[0]
	require.NotNil(t, stale.PendingInterrupt)
	require.Equal(t, "it-stale", stale.PendingInterrupt.ID)
	require.False(t, stale.PendingInterrupt.Actionable, "非 awaiting_input 的中断不可续")

	assistant := list.List[2]
	require.Equal(t, aicontract.MessageStatusAwaitingInput, assistant.Status, "status 必须是字符串枚举")
	require.Equal(t, aicontract.RoleAssistant, assistant.Role)
	require.Equal(t, f.artifactIDs, assistant.ArtifactIDs, "artifactIds 按产出顺序批量回填")
	require.Equal(t, []int64{11, 12}, assistant.MemoryIDs)
	require.Equal(t, []string{"继续分析", "导出报告"}, assistant.Suggestions)
	require.True(t, assistant.Edited)
	require.Equal(t, aicontract.Usage{
		InputTokens: 100, OutputTokens: 40, CachedInputTokens: 20, Credits: 7,
	}, assistant.Usage)
	require.NotEmpty(t, assistant.CreateTime)
	_, parseErr := time.ParseInLocation("2006-01-02T15:04:05", assistant.CreateTime, time.Local)
	require.NoError(t, parseErr, "契约时间为本地 ISO-8601（秒精度，无时区偏移）")

	// blocks 按 seq 升序（思考 → 正文 → 工具），工具调用完整保留
	require.Len(t, assistant.Blocks, 3)
	require.Equal(t, []string{aicontract.BlockKindThinking, aicontract.BlockKindText, aicontract.BlockKindToolUse},
		[]string{assistant.Blocks[0].Kind, assistant.Blocks[1].Kind, assistant.Blocks[2].Kind})
	require.EqualValues(t, 1, assistant.Blocks[0].Seq)
	require.Equal(t, "先读直方图", *assistant.Blocks[0].Text)
	require.Equal(t, "已生成指标报告", *assistant.Blocks[1].Text)
	tool := assistant.Blocks[2].Tool
	require.NotNil(t, tool)
	require.Equal(t, "call-1", tool.CallID)
	require.Equal(t, aicontract.ToolStatusSuccess, tool.Status)
	require.Equal(t, "dehaze", tool.Input["query"])

	require.NotNil(t, assistant.Plan)
	require.Equal(t, aicontract.PlanStatusRunning, assistant.Plan.Status)
	require.True(t, assistant.Plan.AwaitingApproval)
	require.Len(t, assistant.Plan.Tasks, 1)
	require.Len(t, assistant.Plan.Revisions, 1)
	require.Equal(t, "assistant", *assistant.Plan.Revisions[0].Actor)

	require.NotNil(t, assistant.PendingInterrupt)
	require.Equal(t, aicontract.InterruptTypeConfirm, assistant.PendingInterrupt.Type)
	require.True(t, assistant.PendingInterrupt.Actionable, "未过期中断可续")

	// 过期中断（仍 awaiting_input）：payload 保留但 actionable=false
	require.NotNil(t, list.List[1].PendingInterrupt)
	require.Equal(t, "it-expired", list.List[1].PendingInterrupt.ID)
	require.False(t, list.List[1].PendingInterrupt.Actionable, "过期中断不可续")

	// 无块消息：数组字段必须是 []，不能是 null
	empty := list.List[1]
	require.NotNil(t, empty.Blocks)
	require.Empty(t, empty.Blocks)
	require.NotNil(t, empty.ArtifactIDs)
	require.NotNil(t, empty.MemoryIDs)
	require.NotNil(t, empty.Suggestions)
}

// TestGetMessageDerivesContractFields 详情与列表同形状，且不带推理步骤。
func TestGetMessageDerivesContractFields(t *testing.T) {
	f := newAiConversationFixture(t)
	data := decodeData(t, f.do(t, "/ai/messages/"+pathID(f.assistantID)))

	var message aicontract.Message
	require.NoError(t, json.Unmarshal(data, &message))
	require.Equal(t, f.assistantID, message.ID)
	require.Equal(t, f.convID, message.ConversationID)
	require.NotNil(t, message.ParentMessageID)
	require.Equal(t, f.userMsgID, *message.ParentMessageID)
	require.Len(t, message.Blocks, 3)
	require.Equal(t, f.artifactIDs, message.ArtifactIDs)
	require.NotContains(t, string(data), `"thoughts"`)
}

// TestGetMessageInterruptLifecycle 中断可查：未过期为 true，过期为 false，无中断为 null。
func TestGetMessageInterruptLifecycle(t *testing.T) {
	f := newAiConversationFixture(t)

	t.Run("未过期：可续", func(t *testing.T) {
		data := decodeData(t, f.do(t, "/ai/messages/"+pathID(f.assistantID)+"/interrupt"))
		var interrupt aicontract.Interrupt
		require.NoError(t, json.Unmarshal(data, &interrupt))
		require.Equal(t, "it-1", interrupt.ID)
		require.True(t, interrupt.Actionable)
	})

	t.Run("超出 expiresAt：明确标注不可续", func(t *testing.T) {
		data := decodeData(t, f.do(t, "/ai/messages/"+pathID(f.expiredID)+"/interrupt"))
		var interrupt aicontract.Interrupt
		require.NoError(t, json.Unmarshal(data, &interrupt))
		require.Equal(t, "it-expired", interrupt.ID)
		require.False(t, interrupt.Actionable, "过期中断必须 actionable=false")
	})

	t.Run("无待处理中断：data 为 null", func(t *testing.T) {
		data := decodeData(t, f.do(t, "/ai/messages/"+pathID(f.userMsgID)+"/interrupt"))
		require.JSONEq(t, "null", string(data))
	})
}

// TestMessageInterruptActionableUnified 列表/详情/中断查询三处口径一致：
// 过期与非 awaiting_input 的残留中断都返回 actionable=false，且 payload 保留。
func TestMessageInterruptActionableUnified(t *testing.T) {
	f := newAiConversationFixture(t)

	var list aicontract.MessageList
	require.NoError(t, json.Unmarshal(
		decodeData(t, f.do(t, "/ai/conversations/"+pathID(f.convID)+"/messages")), &list))
	interrupts := make(map[int64]*aicontract.Interrupt, len(list.List))
	for i := range list.List {
		interrupts[list.List[i].ID] = list.List[i].PendingInterrupt
	}

	cases := []struct {
		name      string
		msgID     int64
		interrupt string
	}{
		{"过期", f.expiredID, "it-expired"},
		{"非 awaiting_input 残留", f.staleID, "it-stale"},
	}
	for _, tc := range cases {
		t.Run("列表/"+tc.name, func(t *testing.T) {
			requireNotActionable(t, interrupts[tc.msgID], tc.interrupt)
		})

		t.Run("详情/"+tc.name, func(t *testing.T) {
			var message aicontract.Message
			require.NoError(t, json.Unmarshal(
				decodeData(t, f.do(t, "/ai/messages/"+pathID(tc.msgID))), &message))
			requireNotActionable(t, message.PendingInterrupt, tc.interrupt)
		})

		t.Run("中断查询/"+tc.name, func(t *testing.T) {
			var interrupt aicontract.Interrupt
			require.NoError(t, json.Unmarshal(
				decodeData(t, f.do(t, "/ai/messages/"+pathID(tc.msgID)+"/interrupt")), &interrupt))
			requireNotActionable(t, &interrupt, tc.interrupt)
		})
	}
}

func requireNotActionable(t *testing.T, interrupt *aicontract.Interrupt, wantID string) {
	t.Helper()
	require.NotNil(t, interrupt, "中断 payload 必须保留")
	require.Equal(t, wantID, interrupt.ID, "payload 保留")
	require.False(t, interrupt.Actionable, "过期或非 awaiting_input 的中断必须 actionable=false")
}

// TestExportMarkdownDerivedFromBlocks 导出正文来自 blocks 的 text 块（思考/工具不进正文）。
func TestExportMarkdownDerivedFromBlocks(t *testing.T) {
	f := newAiConversationFixture(t)
	rec := f.do(t, "/ai/conversations/"+pathID(f.convID)+"/export?format=markdown")
	body := rec.Body.String()

	require.Contains(t, body, "请分析这张图")
	require.Contains(t, body, "已生成指标报告")
	require.NotContains(t, body, "先读直方图", "思考块不进入导出正文")
	require.NotContains(t, body, "dehaze", "工具入参不进入导出正文")
	require.Contains(t, rec.Header().Get("Content-Disposition"), "conversation_")
}

// TestConversationFeedbackEndpoint 会话级批量反馈：本人视角返回会话内全部反馈，
// 非本人会话按"会话不存在"拒绝（脚本与单条反馈一致）。
func TestConversationFeedbackEndpoint(t *testing.T) {
	db := testutil.NewTestDB(t)
	ctx := context.Background()
	convRepo := aidomainrepo.NewConversationRepository(db)
	fbRepo := aidomainrepo.NewMessageFeedbackRepository(db)
	msgRepo := aidomainrepo.NewMessageRepository(db)
	conversationAPI := NewAiConversationApi(
		nil, nil,
		aidomain.NewFeedbackService(convRepo, msgRepo, fbRepo, aidomainrepo.NewMemoryRepository(db)),
		nil,
	)

	engine := gin.New()
	engine.Use(middleware.ContextErrorHandler())
	engine.Use(func(c *gin.Context) {
		c.Set("claims", &security.CustomClaims{UserID: 910001})
		c.Next()
	})
	engine.GET("/ai/conversations/:id/feedback", conversationAPI.ListConversationFeedback)

	conv := &model.SysAiConversation{UserID: 910001, Title: "批量反馈", TitleSource: "auto", Status: 1}
	require.NoError(t, convRepo.Create(ctx, conv))
	otherConv := &model.SysAiConversation{UserID: 910002, Title: "他人会话", TitleSource: "auto", Status: 1}
	require.NoError(t, convRepo.Create(ctx, otherConv))

	conversationID := conv.ID
	comment := "答非所问"
	for _, fb := range []*model.SysAiMessageFeedback{
		{MessageID: 201, UserID: 910001, ConversationID: &conversationID, Rating: 1, Tags: `["accurate"]`, Source: "internal"},
		{MessageID: 202, UserID: 910001, ConversationID: &conversationID, Rating: -1, Tags: `["irrelevant"]`, Comment: &comment, Source: "internal"},
		{MessageID: 203, UserID: 910002, ConversationID: &conversationID, Rating: 1, Tags: `["accurate"]`, Source: "internal"},
	} {
		require.NoError(t, db.Create(fb).Error)
	}

	rec := httptest.NewRecorder()
	engine.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/ai/conversations/"+pathID(conv.ID)+"/feedback", nil))
	require.Equal(t, http.StatusOK, rec.Code)

	var items []aicontract.Feedback
	require.NoError(t, json.Unmarshal(decodeData(t, rec), &items))
	require.Len(t, items, 2, "只返回当前用户在本会话的反馈")
	require.Equal(t, int64(201), items[0].MessageID)
	require.EqualValues(t, 1, items[0].Rating)
	require.Equal(t, []string{"accurate"}, items[0].Tags)
	require.Equal(t, int64(202), items[1].MessageID)
	require.EqualValues(t, -1, items[1].Rating)
	require.NotNil(t, items[1].Comment)
	require.Equal(t, "答非所问", *items[1].Comment)

	t.Run("非本人会话：会话不存在", func(t *testing.T) {
		rec := httptest.NewRecorder()
		engine.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/ai/conversations/"+pathID(otherConv.ID)+"/feedback", nil))
		require.Contains(t, rec.Body.String(), common.RESOURCE_NOT_FOUND.Code)
	})
}

func pathID(id int64) string { return strconv.FormatInt(id, 10) }

func strPtr(value string) *string { return &value }

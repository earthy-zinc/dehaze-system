package aidomain

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/earthyzinc/clarity-go/internal/model"
	"github.com/earthyzinc/clarity-go/internal/model/aicontract"
	repo "github.com/earthyzinc/clarity-go/internal/repository/aidomain"
	"github.com/earthyzinc/clarity-go/internal/testutil"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// 中断 actionable 口径回归：列表、详情、中断查询三处走同一个 normalizePendingInterrupt。
// 过期（expiresAt 已过）与残留（消息已不是 awaiting_input）都必须是 actionable=false，
// 且 payload（id/type/expiresAt 等）原样保留，供前端展示"不可续"而不是报错。

func newMessageInterruptFixture(t *testing.T, userID int64) (*MessageService, *gorm.DB, int64) {
	t.Helper()
	db := testutil.NewTestDB(t)
	convRepo := repo.NewConversationRepository(db)
	svc := NewMessageService(convRepo, repo.NewMessageRepository(db), repo.NewArtifactRepository(db))
	conv := &model.SysAiConversation{UserID: userID, Title: "中断口径回归", TitleSource: "auto", Status: 1}
	require.NoError(t, convRepo.Create(context.Background(), conv))
	return svc, db, conv.ID
}

// interruptPayload 构造待恢复中断 JSON 列，expiresAt 用 RFC3339 明确时区，避免用例受进程时区影响。
func interruptPayload(t *testing.T, id string, expiresAt time.Time) string {
	t.Helper()
	raw, err := json.Marshal(aicontract.Interrupt{
		ID:         id,
		Type:       aicontract.InterruptTypeConfirm,
		CreatedAt:  time.Now().Add(-time.Minute).Format(time.RFC3339),
		ExpiresAt:  expiresAt.Format(time.RFC3339),
		Actionable: true,
	})
	require.NoError(t, err)
	return string(raw)
}

// insertInterruptMessage 插入一条带中断 payload 的助手消息（parent 串成激活链），返回消息 id。
func insertInterruptMessage(t *testing.T, db *gorm.DB, convID int64, parent *int64, status, payload string) int64 {
	t.Helper()
	msg := &model.SysAiMessage{
		ConversationID: convID, ParentMessageID: parent,
		Role: aicontract.RoleAssistant, Status: status,
		PendingInterrupt: payload,
	}
	require.NoError(t, db.Create(msg).Error)
	return msg.ID
}

// requireNotActionable 断言中断 payload 保留且不可续。
func requireNotActionable(t *testing.T, interrupt *aicontract.Interrupt, wantID string) {
	t.Helper()
	require.NotNil(t, interrupt, "中断 payload 必须保留，不能被吞成 null")
	require.Equal(t, wantID, interrupt.ID, "payload 保留")
	require.NotEmpty(t, interrupt.ExpiresAt, "payload 保留")
	require.False(t, interrupt.Actionable, "过期或非 awaiting_input 的中断必须 actionable=false")
}

// TestMessageInterruptActionableUnifiedAcrossReadPaths 三处读取对两种输入口径一致。
func TestMessageInterruptActionableUnifiedAcrossReadPaths(t *testing.T) {
	const userID = 780001
	svc, db, convID := newMessageInterruptFixture(t, userID)
	ctx := context.Background()

	now := time.Now()
	expiredID := insertInterruptMessage(t, db, convID, nil, aicontract.MessageStatusAwaitingInput,
		interruptPayload(t, "it-expired", now.Add(-time.Minute)))
	staleID := insertInterruptMessage(t, db, convID, &expiredID, aicontract.MessageStatusCompleted,
		interruptPayload(t, "it-stale", now.Add(time.Hour)))
	clearID := insertInterruptMessage(t, db, convID, &staleID, aicontract.MessageStatusCompleted, "")

	// 列表：一次查询覆盖三种消息
	list, err := svc.List(ctx, convID, userID, nil, 50, false)
	require.NoError(t, err)
	interrupts := make(map[int64]*aicontract.Interrupt, len(list.List))
	for i := range list.List {
		interrupts[list.List[i].ID] = list.List[i].PendingInterrupt
	}
	requireNotActionable(t, interrupts[expiredID], "it-expired")
	requireNotActionable(t, interrupts[staleID], "it-stale")
	require.Nil(t, interrupts[clearID], "无中断 payload 的消息不下发 pendingInterrupt")

	// 详情与中断查询：逐个消息走各自的读取路径
	cases := []struct {
		name      string
		msgID     int64
		interrupt string
	}{
		{"过期", expiredID, "it-expired"},
		{"非 awaiting_input 残留", staleID, "it-stale"},
	}
	for _, tc := range cases {
		t.Run("详情/"+tc.name, func(t *testing.T) {
			msg, err := svc.Get(ctx, tc.msgID, userID, false)
			require.NoError(t, err)
			require.Equal(t, tc.msgID, msg.ID)
			requireNotActionable(t, msg.PendingInterrupt, tc.interrupt)
		})

		t.Run("中断查询/"+tc.name, func(t *testing.T) {
			interrupt, err := svc.GetInterrupt(ctx, tc.msgID, userID, false)
			require.NoError(t, err)
			requireNotActionable(t, interrupt, tc.interrupt)
		})
	}

	t.Run("中断查询/无 payload 返回 nil", func(t *testing.T) {
		interrupt, err := svc.GetInterrupt(ctx, clearID, userID, false)
		require.NoError(t, err)
		require.Nil(t, interrupt)
	})
}

// TestMessageInterruptActionableKeepsLiveInterrupt 未过期且 awaiting_input 的中断三处都可续。
func TestMessageInterruptActionableKeepsLiveInterrupt(t *testing.T) {
	const userID = 780002
	svc, db, convID := newMessageInterruptFixture(t, userID)
	ctx := context.Background()

	liveID := insertInterruptMessage(t, db, convID, nil, aicontract.MessageStatusAwaitingInput,
		interruptPayload(t, "it-live", time.Now().Add(time.Hour)))

	list, err := svc.List(ctx, convID, userID, nil, 50, false)
	require.NoError(t, err)
	require.Len(t, list.List, 1)
	require.NotNil(t, list.List[0].PendingInterrupt)
	require.True(t, list.List[0].PendingInterrupt.Actionable, "未过期中断在列表里可续")

	msg, err := svc.Get(ctx, liveID, userID, false)
	require.NoError(t, err)
	require.True(t, msg.PendingInterrupt.Actionable, "未过期中断在详情里可续")

	interrupt, err := svc.GetInterrupt(ctx, liveID, userID, false)
	require.NoError(t, err)
	require.True(t, interrupt.Actionable, "未过期中断在中断查询里可续")
}

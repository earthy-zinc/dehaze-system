package aidomain

import (
	"context"
	"testing"
	"time"

	"github.com/earthyzinc/clarity-go/internal/model"
	"github.com/earthyzinc/clarity-go/internal/model/aicontract"
	"github.com/earthyzinc/clarity-go/internal/testutil"
	"github.com/stretchr/testify/require"
)

// 消息新契约的持久化回归：blocks/plan/pending_interrupt/suggestions 四种 JSON 列按契约形状
// 读写，status 为字符串枚举，旧 content/tool_calls/tool_call_id/metadata 列已不存在。

func newMessageContractFixture(t *testing.T, userID int64) (*MessageRepository, *ConversationRepository, int64) {
	t.Helper()
	db := testutil.NewTestDB(t)
	convRepo := NewConversationRepository(db)
	conv := &model.SysAiConversation{UserID: userID, Title: "消息契约回归", TitleSource: "auto", Status: 1}
	require.NoError(t, convRepo.Create(context.Background(), conv))
	return NewMessageRepository(db), convRepo, conv.ID
}

// TestMessageBlocksRoundTrip blocks 按契约形状落库并原样读回（seq/kind/tool 均不丢字段）。
func TestMessageBlocksRoundTrip(t *testing.T) {
	repo, _, convID := newMessageContractFixture(t, 880001)
	ctx := context.Background()

	tool := &aicontract.ToolCall{
		CallID: "call-9", Name: "image_dehaze", Input: map[string]any{"strength": 0.6},
		Status: aicontract.ToolStatusSuccess, SubAgent: true,
	}
	msg := &model.SysAiMessage{
		ConversationID: convID, Role: aicontract.RoleAssistant,
		Status: aicontract.MessageStatusCompleted,
		Blocks: aicontract.EncodeBlocks([]aicontract.Block{
			{ID: "b2", Kind: aicontract.BlockKindText, Seq: 2, Text: strPtr("处理完成")},
			{ID: "b1", Kind: aicontract.BlockKindToolUse, Seq: 1, Tool: tool},
		}),
		Suggestions:   `["继续"]`,
		UsedMemoryIDs: `[3,4]`,
		Edited:        1,
	}
	require.NoError(t, repo.Create(ctx, msg))

	loaded, err := repo.GetByID(ctx, msg.ID)
	require.NoError(t, err)
	blocks := aicontract.DecodeBlocks(loaded.Blocks)
	require.Len(t, blocks, 2)
	require.Equal(t, "b1", blocks[0].ID, "seq 小的块在前")
	require.Equal(t, aicontract.BlockKindToolUse, blocks[0].Kind)
	require.Equal(t, "image_dehaze", blocks[0].Tool.Name)
	require.Equal(t, aicontract.ToolStatusSuccess, blocks[0].Tool.Status)
	require.True(t, blocks[0].Tool.SubAgent)
	require.Equal(t, "处理完成", aicontract.DeriveText(blocks), "派生文本只取 text 块")
	require.Equal(t, []int64{3, 4}, aicontract.DecodeIDList(loaded.UsedMemoryIDs))
	require.Equal(t, []string{"继续"}, aicontract.DecodeSuggestions(loaded.Suggestions))
	require.Equal(t, aicontract.MessageStatusCompleted, loaded.Status)
	require.Equal(t, 1, loaded.Edited)
}

// TestMessageUpdateFieldsKeepsContractShape 状态/中断的增量更新走 map 更新，清空即落 NULL。
func TestMessageUpdateFieldsKeepsContractShape(t *testing.T) {
	repo, _, convID := newMessageContractFixture(t, 880002)
	ctx := context.Background()

	msg := &model.SysAiMessage{
		ConversationID: convID, Role: aicontract.RoleAssistant,
		Status: aicontract.MessageStatusRunning,
		Blocks: aicontract.EncodeBlocks([]aicontract.Block{
			{ID: "b1", Kind: aicontract.BlockKindText, Seq: 1, Text: strPtr("思考中")},
		}),
	}
	require.NoError(t, repo.Create(ctx, msg))

	expiresAt := time.Now().Add(time.Minute).Format(time.RFC3339)
	require.NoError(t, repo.UpdateFields(ctx, msg.ID, map[string]any{
		"status":            aicontract.MessageStatusAwaitingInput,
		"pending_interrupt": `{"id":"it-2","type":"confirm","createdAt":"` + time.Now().Format(time.RFC3339) + `","expiresAt":"` + expiresAt + `","actionable":true}`,
		"plan":              `{"status":"pending","tasks":[],"revisions":[],"awaitingApproval":false}`,
	}))

	loaded, err := repo.GetByID(ctx, msg.ID)
	require.NoError(t, err)
	require.Equal(t, aicontract.MessageStatusAwaitingInput, loaded.Status, "status 按字符串枚举更新")
	require.NotNil(t, aicontract.DecodeInterrupt(loaded.PendingInterrupt))
	require.NotNil(t, aicontract.DecodePlan(loaded.Plan))

	require.NoError(t, repo.UpdateFields(ctx, msg.ID, map[string]any{
		"status":            aicontract.MessageStatusCompleted,
		"pending_interrupt": nil,
	}))
	loaded, err = repo.GetByID(ctx, msg.ID)
	require.NoError(t, err)
	require.Nil(t, aicontract.DecodeInterrupt(loaded.PendingInterrupt), "清空中断必须落 SQL NULL")
}

// TestListAnomalyStatusByConversationsUsesStringEnum 异常汇总按 failed/canceled 字符串状态聚合。
func TestListAnomalyStatusByConversationsUsesStringEnum(t *testing.T) {
	repo, _, convID := newMessageContractFixture(t, 880003)
	ctx := context.Background()

	for _, status := range []string{
		aicontract.MessageStatusFailed,
		aicontract.MessageStatusCanceled,
		aicontract.MessageStatusCompleted,
	} {
		require.NoError(t, repo.Create(ctx, &model.SysAiMessage{
			ConversationID: convID, Role: aicontract.RoleAssistant, Status: status,
		}))
	}

	statuses, err := repo.ListAnomalyStatusByConversations(ctx, []int64{convID})
	require.NoError(t, err)
	require.Len(t, statuses[convID], 2, "只聚合 failed/canceled 两种终态")
	_, failed := statuses[convID][aicontract.MessageStatusFailed]
	_, canceled := statuses[convID][aicontract.MessageStatusCanceled]
	require.True(t, failed)
	require.True(t, canceled)
}

func strPtr(value string) *string { return &value }

// TestKeywordSearchMatchesBlocksText 关键词检索走 blocks 派生文本（content 列已删除）：
// 标题不命中的会话，靠消息正文命中；命中消息 ID 供审计视角回填。
func TestKeywordSearchMatchesBlocksText(t *testing.T) {
	repo, convRepo, convID := newMessageContractFixture(t, 880004)
	ctx := context.Background()

	require.NoError(t, repo.Create(ctx, &model.SysAiMessage{
		ConversationID: convID, Role: aicontract.RoleAssistant,
		Status: aicontract.MessageStatusCompleted,
		Blocks: aicontract.EncodeBlocks([]aicontract.Block{
			{ID: "b1", Kind: aicontract.BlockKindText, Seq: 1, Text: strPtr("推荐使用去雾算法，参数默认即可")},
		}),
	}))

	convs, total, err := convRepo.PaginateWithKeyword(ctx, 880004, 1, 10, "去雾算法", nil, false)
	require.NoError(t, err)
	require.EqualValues(t, 1, total)
	require.Len(t, convs, 1)
	require.Equal(t, convID, convs[0].ID)

	matched, err := convRepo.ListMatchedMessageIDs(ctx, []int64{convID}, "去雾算法")
	require.NoError(t, err)
	require.Contains(t, matched, convID, "命中会话应回填 matchedMessageId")

	_, total, err = convRepo.PaginateWithKeyword(ctx, 880004, 1, 10, "不存在的关键词", nil, false)
	require.NoError(t, err)
	require.EqualValues(t, 0, total, "正文未命中时不得误匹配")
}

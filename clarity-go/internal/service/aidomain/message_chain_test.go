package aidomain

import (
	"context"
	"testing"

	"github.com/earthyzinc/clarity-go/internal/model"
	"github.com/earthyzinc/clarity-go/internal/model/aicontract"
	repo "github.com/earthyzinc/clarity-go/internal/repository/aidomain"
	"github.com/earthyzinc/clarity-go/internal/testutil"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// 会话激活链（self 视角）回归：列表只返回 current_branch_message_id 回溯出的链，
// siblingIndex/siblingCount 按同父未删 assistant 兄弟批量回填；admin 视角保持全量分支分页。

type messageChainFixture struct {
	service *MessageService
	convs   *repo.ConversationRepository
	msgs    *repo.MessageRepository
	db      *gorm.DB
	convID  int64
	userID  int64
}

func newMessageChainFixture(t *testing.T, userID int64) *messageChainFixture {
	t.Helper()
	db := testutil.NewTestDB(t)
	convRepo := repo.NewConversationRepository(db)
	msgRepo := repo.NewMessageRepository(db)
	conv := &model.SysAiConversation{UserID: userID, Title: "激活链回归", TitleSource: "auto", Status: 1}
	require.NoError(t, convRepo.Create(context.Background(), conv))
	return &messageChainFixture{
		service: NewMessageService(convRepo, msgRepo, repo.NewArtifactRepository(db)),
		convs:   convRepo,
		msgs:    msgRepo,
		db:      db,
		convID:  conv.ID,
		userID:  userID,
	}
}

func (f *messageChainFixture) add(t *testing.T, parent *int64, role string) *model.SysAiMessage {
	t.Helper()
	msg := &model.SysAiMessage{
		ConversationID: f.convID, ParentMessageID: parent,
		Role: role, Status: aicontract.MessageStatusCompleted,
	}
	require.NoError(t, f.db.Create(msg).Error)
	return msg
}

func (f *messageChainFixture) setBranch(t *testing.T, msgID int64) {
	t.Helper()
	require.NoError(t, f.convs.UpdateCurrentBranch(context.Background(), f.convID, msgID, f.userID))
}

// setInterrupt 覆盖消息状态与 pending_interrupt，模拟推理侧落库的中断事实。
func (f *messageChainFixture) setInterrupt(t *testing.T, msgID int64, status, payload string) {
	t.Helper()
	require.NoError(t, f.db.Model(&model.SysAiMessage{}).Where("id = ?", msgID).
		Updates(map[string]any{"status": status, "pending_interrupt": payload}).Error)
}

func (f *messageChainFixture) list(t *testing.T, before *int64, limit int, admin bool) *aicontract.MessageList {
	t.Helper()
	res, err := f.service.List(context.Background(), f.convID, f.userID, before, limit, admin)
	require.NoError(t, err)
	return res
}

// TestMessageListSelfReturnsActiveChain 激活链之外的兄弟分支不进入列表，total 为链长。
func TestMessageListSelfReturnsActiveChain(t *testing.T) {
	f := newMessageChainFixture(t, 8101)
	userMsg := f.add(t, nil, aicontract.RoleUser)
	f.add(t, &userMsg.ID, aicontract.RoleAssistant)
	active := f.add(t, &userMsg.ID, aicontract.RoleAssistant)
	f.add(t, &userMsg.ID, aicontract.RoleAssistant)
	f.setBranch(t, active.ID)

	res := f.list(t, nil, 50, false)
	require.Equal(t, []int64{active.ID, userMsg.ID}, idsOf(res.List), "只返回激活链（id 倒序）")
	require.EqualValues(t, 2, res.Total, "total 为链长而非会话消息总数")
	require.False(t, res.HasMore)

	require.NotNil(t, res.List[0].SiblingIndex)
	require.EqualValues(t, 2, *res.List[0].SiblingIndex, "兄弟序号 1 起，按 id 升序")
	require.NotNil(t, res.List[0].SiblingCount)
	require.EqualValues(t, 3, *res.List[0].SiblingCount)
	// 非 assistant（根级 user）不参与兄弟序号
	require.Nil(t, res.List[1].SiblingIndex)
	require.Nil(t, res.List[1].SiblingCount)
}

// TestMessageListSelfRootAssistantHasNoSiblings 根级 assistant 无父消息，兄弟字段为 null。
func TestMessageListSelfRootAssistantHasNoSiblings(t *testing.T) {
	f := newMessageChainFixture(t, 8102)
	root := f.add(t, nil, aicontract.RoleAssistant)
	f.setBranch(t, root.ID)

	res := f.list(t, nil, 50, false)
	require.Equal(t, []int64{root.ID}, idsOf(res.List))
	require.Nil(t, res.List[0].SiblingIndex)
	require.Nil(t, res.List[0].SiblingCount)
}

// TestMessageListSelfBranchPointerFallback 指针为空/悬空/指向已删消息/跨会话时回退到最大未删消息。
func TestMessageListSelfBranchPointerFallback(t *testing.T) {
	ctx := context.Background()

	// buildChain 造一条 u1 → a1 → u2 → a2 的链，返回按 id 升序的 id 列表。
	buildChain := func(t *testing.T) (*messageChainFixture, []int64) {
		t.Helper()
		f := newMessageChainFixture(t, 8103)
		u1 := f.add(t, nil, aicontract.RoleUser)
		a1 := f.add(t, &u1.ID, aicontract.RoleAssistant)
		u2 := f.add(t, &a1.ID, aicontract.RoleUser)
		a2 := f.add(t, &u2.ID, aicontract.RoleAssistant)
		return f, []int64{u1.ID, a1.ID, u2.ID, a2.ID}
	}

	t.Run("指针为空", func(t *testing.T) {
		f, ids := buildChain(t)
		res := f.list(t, nil, 50, false)
		require.Equal(t, reversed(ids), idsOf(res.List))
		require.EqualValues(t, 4, res.Total)
	})

	t.Run("指针悬空", func(t *testing.T) {
		f, ids := buildChain(t)
		f.setBranch(t, ids[3]+999)
		res := f.list(t, nil, 50, false)
		require.Equal(t, reversed(ids), idsOf(res.List))
	})

	t.Run("指针指向已软删消息", func(t *testing.T) {
		f, ids := buildChain(t)
		require.NoError(t, f.msgs.SoftDeleteByIDs(ctx, []int64{ids[3]}, f.userID))
		f.setBranch(t, ids[3])
		res := f.list(t, nil, 50, false)
		require.Equal(t, []int64{ids[2], ids[1], ids[0]}, idsOf(res.List), "回退到最大未删消息 u2 并回溯")
		require.EqualValues(t, 3, res.Total)
	})

	t.Run("指针指向其他会话的消息", func(t *testing.T) {
		f, ids := buildChain(t)
		otherConv := &model.SysAiConversation{UserID: f.userID, Title: "其他会话", TitleSource: "auto", Status: 1}
		require.NoError(t, f.convs.Create(ctx, otherConv))
		foreign := &model.SysAiMessage{
			ConversationID: otherConv.ID, Role: aicontract.RoleUser, Status: aicontract.MessageStatusCompleted,
		}
		require.NoError(t, f.db.Create(foreign).Error)
		f.setBranch(t, foreign.ID)
		res := f.list(t, nil, 50, false)
		require.Equal(t, reversed(ids), idsOf(res.List))
	})
}

// TestMessageListSelfSoftDeletedAncestorBreaksChain 链中软删消息阻断回溯：只保留断点之后的消息。
func TestMessageListSelfSoftDeletedAncestorBreaksChain(t *testing.T) {
	f := newMessageChainFixture(t, 8104)
	u1 := f.add(t, nil, aicontract.RoleUser)
	a1 := f.add(t, &u1.ID, aicontract.RoleAssistant)
	u2 := f.add(t, &a1.ID, aicontract.RoleUser)
	a2 := f.add(t, &u2.ID, aicontract.RoleAssistant)
	f.setBranch(t, a2.ID)
	require.NoError(t, f.msgs.SoftDeleteByIDs(context.Background(), []int64{a1.ID}, f.userID))

	res := f.list(t, nil, 50, false)
	require.Equal(t, []int64{a2.ID, u2.ID}, idsOf(res.List), "断点之前的消息不可达")
	require.EqualValues(t, 2, res.Total)
}

// TestMessageListSelfPaginationOnChain 分页在链上切：before 之前的最后 limit 条按 id 倒序，hasMore 由链长判断。
func TestMessageListSelfPaginationOnChain(t *testing.T) {
	f := newMessageChainFixture(t, 8105)
	ids := make([]int64, 0, 5)
	var parent *int64
	for i := 0; i < 5; i++ {
		role := aicontract.RoleUser
		if i%2 == 1 {
			role = aicontract.RoleAssistant
		}
		msg := f.add(t, parent, role)
		ids = append(ids, msg.ID)
		parent = &msg.ID
	}
	// 链外分支：挂在第 2 条消息下，任何一页都不得出现
	offChain := f.add(t, &ids[1], aicontract.RoleAssistant)
	f.setBranch(t, ids[4])

	page1 := f.list(t, nil, 2, false)
	require.Equal(t, []int64{ids[4], ids[3]}, idsOf(page1.List))
	require.True(t, page1.HasMore)
	require.EqualValues(t, 5, page1.Total, "total 恒为链长，不受分页影响")

	page2 := f.list(t, &ids[3], 2, false)
	require.Equal(t, []int64{ids[2], ids[1]}, idsOf(page2.List))
	require.True(t, page2.HasMore)

	page3 := f.list(t, &ids[1], 2, false)
	require.Equal(t, []int64{ids[0]}, idsOf(page3.List))
	require.False(t, page3.HasMore)
	require.EqualValues(t, 5, page3.Total)

	for _, page := range []*aicontract.MessageList{page1, page2, page3} {
		require.NotContains(t, idsOf(page.List), offChain.ID, "激活链之外的兄弟分支不得混入")
	}
}

// TestMessageListSiblingPositionSkipsDeleted 兄弟序号按未删兄弟计算，软删兄弟不占位。
func TestMessageListSiblingPositionSkipsDeleted(t *testing.T) {
	f := newMessageChainFixture(t, 8106)
	userMsg := f.add(t, nil, aicontract.RoleUser)
	deleted := f.add(t, &userMsg.ID, aicontract.RoleAssistant)
	f.add(t, &userMsg.ID, aicontract.RoleAssistant)
	active := f.add(t, &userMsg.ID, aicontract.RoleAssistant)
	f.add(t, &userMsg.ID, aicontract.RoleAssistant)
	f.setBranch(t, active.ID)
	require.NoError(t, f.msgs.SoftDeleteByIDs(context.Background(), []int64{deleted.ID}, f.userID))

	res := f.list(t, nil, 1, false)
	require.Equal(t, []int64{active.ID}, idsOf(res.List))
	require.EqualValues(t, 2, *res.List[0].SiblingIndex, "软删兄弟不占位，序号从 1 连续")
	require.EqualValues(t, 3, *res.List[0].SiblingCount, "软删兄弟不计入总数")
	require.True(t, res.HasMore, "链上仍有更早消息")
	require.EqualValues(t, 2, res.Total)
}

// TestMessageListAdminKeepsFullBranches admin 视角仍返回全部分支，total 为会话消息总数，兄弟字段为 null。
func TestMessageListAdminKeepsFullBranches(t *testing.T) {
	f := newMessageChainFixture(t, 8107)
	userMsg := f.add(t, nil, aicontract.RoleUser)
	a1 := f.add(t, &userMsg.ID, aicontract.RoleAssistant)
	a2 := f.add(t, &userMsg.ID, aicontract.RoleAssistant)
	a3 := f.add(t, &userMsg.ID, aicontract.RoleAssistant)
	f.setBranch(t, a1.ID)

	res, err := f.service.List(context.Background(), f.convID, 999999, nil, 50, true)
	require.NoError(t, err)
	require.Equal(t, []int64{a3.ID, a2.ID, a1.ID, userMsg.ID}, idsOf(res.List))
	require.EqualValues(t, 4, res.Total, "admin total 为会话消息总数")
	require.False(t, res.HasMore)
	for i := range res.List {
		require.Nil(t, res.List[i].SiblingIndex, "admin 视角不下发兄弟字段")
		require.Nil(t, res.List[i].SiblingCount)
	}

	t.Run("admin 分页仍按全量消息切", func(t *testing.T) {
		page, err := f.service.List(context.Background(), f.convID, 999999, nil, 2, true)
		require.NoError(t, err)
		require.Equal(t, []int64{a3.ID, a2.ID}, idsOf(page.List))
		require.True(t, page.HasMore)
		require.EqualValues(t, 4, page.Total)
	})
}

// TestMessageListSelfEmptyConversation 空会话返回空链。
func TestMessageListSelfEmptyConversation(t *testing.T) {
	f := newMessageChainFixture(t, 8108)
	res := f.list(t, nil, 50, false)
	require.Empty(t, res.List)
	require.EqualValues(t, 0, res.Total)
	require.False(t, res.HasMore)
}

// TestMessageListSelfRejectsForeignUser 非归属用户 self 视角不可见。
func TestMessageListSelfRejectsForeignUser(t *testing.T) {
	f := newMessageChainFixture(t, 8109)
	f.add(t, nil, aicontract.RoleUser)
	_, err := f.service.List(context.Background(), f.convID, 999999, nil, 50, false)
	require.Error(t, err)
}

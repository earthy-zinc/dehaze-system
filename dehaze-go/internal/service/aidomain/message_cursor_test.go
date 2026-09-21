package aidomain

import (
	"context"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	repo "github.com/earthyzinc/dehaze-go/internal/repository/aidomain"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// 消息列表游标分页（before/limit）行为回归：契约见三端定稿——
// GET /conversations/{id}/messages?before=&limit=，排序 id DESC，响应 {list,total,hasMore}。

func newMessageCursorFixture(t *testing.T, userID int64) (*MessageService, *gorm.DB, int64) {
	t.Helper()
	db := testutil.NewTestDB(t)
	convRepo := repo.NewConversationRepository(db)
	svc := NewMessageService(convRepo, repo.NewMessageRepository(db), repo.NewThoughtRepository(db))
	conv := &model.SysAiConversation{UserID: userID, Title: "游标分页回归", TitleSource: "auto", Status: 1}
	require.NoError(t, convRepo.Create(context.Background(), conv))
	return svc, db, conv.ID
}

// insertMessages 顺序插入 count 条用户消息，返回按插入顺序（id 升序）的 id 列表。
func insertMessages(t *testing.T, db *gorm.DB, convID int64, count int) []int64 {
	t.Helper()
	ids := make([]int64, 0, count)
	for i := 0; i < count; i++ {
		msg := &model.SysAiMessage{ConversationID: convID, Role: "user", Status: 1}
		require.NoError(t, db.Create(msg).Error)
		ids = append(ids, msg.ID)
	}
	return ids
}

func reversed(ids []int64) []int64 {
	out := make([]int64, len(ids))
	for i, id := range ids {
		out[len(ids)-1-i] = id
	}
	return out
}

func idsOf(list []MessageVO) []int64 {
	out := make([]int64, 0, len(list))
	for i := range list {
		out = append(out, list[i].ID)
	}
	return out
}

// TestMessageListCursorDefaultLatestPage before 缺省取最新一页，id 倒序，total 为该会话总数。
func TestMessageListCursorDefaultLatestPage(t *testing.T) {
	svc, db, convID := newMessageCursorFixture(t, 7001)
	ids := insertMessages(t, db, convID, 5)

	res, err := svc.List(context.Background(), convID, 7001, nil, 50, false)
	require.NoError(t, err)
	require.EqualValues(t, 5, res.Total, "total 为该会话消息总数")
	require.False(t, res.HasMore, "消息数未超过 limit，hasMore 应为 false")
	require.Equal(t, reversed(ids), idsOf(res.List), "缺省 before 取最新一页（id 倒序）")
}

// TestMessageListCursorBeforeBoundary before 边界：=最小 id / 不存在的 id / 中间 id。
func TestMessageListCursorBeforeBoundary(t *testing.T) {
	svc, db, convID := newMessageCursorFixture(t, 7002)
	ids := insertMessages(t, db, convID, 4)
	ctx := context.Background()

	t.Run("before=最小id：无更早消息", func(t *testing.T) {
		res, err := svc.List(ctx, convID, 7002, &ids[0], 50, false)
		require.NoError(t, err)
		require.Empty(t, res.List)
		require.False(t, res.HasMore)
		require.EqualValues(t, 4, res.Total)
	})

	t.Run("before=不存在的id（大于最大id）：取全部", func(t *testing.T) {
		absent := ids[len(ids)-1] + 5000
		res, err := svc.List(ctx, convID, 7002, &absent, 50, false)
		require.NoError(t, err)
		require.Equal(t, reversed(ids), idsOf(res.List))
		require.False(t, res.HasMore)
	})

	t.Run("before=中间id：仅取更早消息", func(t *testing.T) {
		mid := ids[1] + 1
		res, err := svc.List(ctx, convID, 7002, &mid, 50, false)
		require.NoError(t, err)
		require.Equal(t, []int64{ids[1], ids[0]}, idsOf(res.List))
	})
}

// TestMessageListCursorHasMoreProbe hasMore 用 limit+1 探测：真假两态与逐页翻到底。
func TestMessageListCursorHasMoreProbe(t *testing.T) {
	svc, db, convID := newMessageCursorFixture(t, 7003)
	ids := insertMessages(t, db, convID, 5)
	ctx := context.Background()

	page1, err := svc.List(ctx, convID, 7003, nil, 3, false)
	require.NoError(t, err)
	require.Len(t, page1.List, 3)
	require.True(t, page1.HasMore, "还有更早消息时 hasMore=true")
	require.EqualValues(t, 5, page1.Total)
	require.Equal(t, []int64{ids[4], ids[3], ids[2]}, idsOf(page1.List))

	last := page1.List[len(page1.List)-1].ID
	page2, err := svc.List(ctx, convID, 7003, &last, 3, false)
	require.NoError(t, err)
	require.Len(t, page2.List, 2)
	require.False(t, page2.HasMore, "最后一页 hasMore=false")
	require.Equal(t, []int64{ids[1], ids[0]}, idsOf(page2.List))

	t.Run("limit=1 恰好一条", func(t *testing.T) {
		res, err := svc.List(ctx, convID, 7003, nil, 1, false)
		require.NoError(t, err)
		require.Len(t, res.List, 1)
		require.True(t, res.HasMore)
		require.Equal(t, ids[4], res.List[0].ID)
	})
}

// TestMessageListCursorEmptyConversation 空会话：空列表、total=0、hasMore=false。
func TestMessageListCursorEmptyConversation(t *testing.T) {
	svc, _, convID := newMessageCursorFixture(t, 7004)

	res, err := svc.List(context.Background(), convID, 7004, nil, 50, false)
	require.NoError(t, err)
	require.Empty(t, res.List)
	require.EqualValues(t, 0, res.Total)
	require.False(t, res.HasMore)
}

// TestMessageListCursorViewAdmin view=admin 视角：跨用户可见，非 admin 越权不可见（保持既有权限逻辑）。
func TestMessageListCursorViewAdmin(t *testing.T) {
	svc, db, convID := newMessageCursorFixture(t, 7005)
	ids := insertMessages(t, db, convID, 2)
	ctx := context.Background()

	t.Run("admin=true 跨用户可见", func(t *testing.T) {
		res, err := svc.List(ctx, convID, 999999, nil, 50, true)
		require.NoError(t, err)
		require.Equal(t, reversed(ids), idsOf(res.List))
	})

	t.Run("admin=false 非归属用户不可见", func(t *testing.T) {
		_, err := svc.List(ctx, convID, 999999, nil, 50, false)
		require.Error(t, err)
	})
}

// TestMessageListCursorConcurrentInsertNoDupNoMiss 并发插入下按 before 翻页不重不漏：
// 快照起点取 anchor=maxOriginal+1，翻页期间另一 goroutine 持续写入更大 id 的新消息，
// 全量遍历必须恰好覆盖快照内的原始消息（新消息因 id >= anchor 被排除，不重不漏）。
func TestMessageListCursorConcurrentInsertNoDupNoMiss(t *testing.T) {
	db := testutil.NewPoolTestDB(t)
	ctx := context.Background()
	convRepo := repo.NewConversationRepository(db)
	svc := NewMessageService(convRepo, repo.NewMessageRepository(db), repo.NewThoughtRepository(db))

	const userID = 770001
	conv := &model.SysAiConversation{UserID: userID, Title: "游标并发翻页", TitleSource: "auto", Status: 1}
	require.NoError(t, convRepo.Create(ctx, conv))
	t.Cleanup(func() {
		_ = db.Exec("DELETE FROM sys_ai_message WHERE conversation_id = ?", conv.ID).Error
		_ = db.Exec("DELETE FROM sys_ai_conversation WHERE id = ?", conv.ID).Error
	})

	original := insertMessages(t, db, conv.ID, 30)
	expected := reversed(original)
	anchor := original[len(original)-1] + 1

	var wg sync.WaitGroup
	insertErrs := make(chan error, 1)
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 16; i++ {
			msg := &model.SysAiMessage{ConversationID: conv.ID, Role: "user", Status: 1}
			if err := db.Create(msg).Error; err != nil {
				insertErrs <- err
				return
			}
			time.Sleep(2 * time.Millisecond)
		}
	}()

	got := make([]int64, 0, len(original))
	before := &anchor
	for {
		res, err := svc.List(ctx, conv.ID, userID, before, 7, false)
		require.NoError(t, err)
		got = append(got, idsOf(res.List)...)
		if !res.HasMore {
			break
		}
		last := res.List[len(res.List)-1].ID
		before = &last
		time.Sleep(time.Millisecond)
	}

	wg.Wait()
	close(insertErrs)
	for err := range insertErrs {
		require.NoError(t, err)
	}

	require.Equal(t, expected, got, "按 before 翻页必须覆盖快照内全部消息且不重不漏")
	require.Len(t, sortDedup(got), len(got), "结果中不得出现重复 id")
}

func sortDedup(ids []int64) []int64 {
	seen := make(map[int64]struct{}, len(ids))
	out := make([]int64, 0, len(ids))
	for _, id := range ids {
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

package aidomain

import (
	"context"
	"testing"

	"github.com/earthyzinc/clarity-go/internal/model"
	repo "github.com/earthyzinc/clarity-go/internal/repository/aidomain"
	"github.com/earthyzinc/clarity-go/internal/testutil"
	"github.com/earthyzinc/clarity-go/pkg/common"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// 会话级批量反馈：一次查询返回当前用户在该会话下的全部未删反馈，供进入会话时批量回填。

func newFeedbackListFixture(t *testing.T, userID int64, db *gorm.DB) (*FeedbackService, *repo.ConversationRepository, int64) {
	t.Helper()
	convRepo := repo.NewConversationRepository(db)
	svc := NewFeedbackService(
		convRepo,
		repo.NewMessageRepository(db),
		repo.NewMessageFeedbackRepository(db),
		repo.NewMemoryRepository(db),
	)
	conv := &model.SysAiConversation{UserID: userID, Title: "批量反馈", TitleSource: "auto", Status: 1}
	require.NoError(t, convRepo.Create(context.Background(), conv))
	return svc, convRepo, conv.ID
}

func insertFeedback(t *testing.T, db *gorm.DB, convID, messageID, userID int64, rating int, tags string, comment *string, deleted int64) *model.SysAiMessageFeedback {
	t.Helper()
	fb := &model.SysAiMessageFeedback{
		MessageID: messageID, UserID: userID, ConversationID: &convID,
		Rating: rating, Tags: tags, Comment: comment, Source: "internal", Deleted: deleted,
	}
	require.NoError(t, db.Create(fb).Error)
	return fb
}

// TestFeedbackListByConversation 只返回本人 + 本会话 + 未删反馈，按消息 id 升序。
func TestFeedbackListByConversation(t *testing.T) {
	db := testutil.NewTestDB(t)
	ctx := context.Background()
	const userID = 8201
	svc, convRepo, convID := newFeedbackListFixture(t, userID, db)

	otherConv := &model.SysAiConversation{UserID: userID, Title: "其他会话", TitleSource: "auto", Status: 1}
	require.NoError(t, convRepo.Create(ctx, otherConv))

	comment := "太啰嗦了"
	first := insertFeedback(t, db, convID, 101, userID, 1, `["accurate"]`, nil, 0)
	second := insertFeedback(t, db, convID, 102, userID, -1, `["too_long"]`, &comment, 0)
	insertFeedback(t, db, otherConv.ID, 103, userID, 1, `["concise"]`, nil, 0) // 其他会话
	insertFeedback(t, db, convID, 104, 8202, 1, `["accurate"]`, nil, 0)        // 其他用户
	insertFeedback(t, db, convID, 105, userID, 1, `["creative"]`, nil, 105)    // 已撤销

	items, err := svc.ListByConversation(ctx, convID, userID)
	require.NoError(t, err)
	require.Len(t, items, 2)

	require.Equal(t, first.ID, *items[0].ID)
	require.Equal(t, int64(101), items[0].MessageID)
	require.Equal(t, int64(userID), items[0].UserID)
	require.NotNil(t, items[0].ConversationID)
	require.Equal(t, convID, *items[0].ConversationID)
	require.EqualValues(t, 1, items[0].Rating)
	require.Equal(t, []string{"accurate"}, items[0].Tags)
	require.Nil(t, items[0].Comment)
	require.NotEmpty(t, *items[0].CreateTime)

	require.Equal(t, second.ID, *items[1].ID)
	require.EqualValues(t, -1, items[1].Rating)
	require.Equal(t, []string{"too_long"}, items[1].Tags)
	require.NotNil(t, items[1].Comment)
	require.Equal(t, "太啰嗦了", *items[1].Comment)
}

// TestFeedbackListByConversationEmpty 无反馈返回空数组而非 null。
func TestFeedbackListByConversationEmpty(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc, _, convID := newFeedbackListFixture(t, 8203, db)

	items, err := svc.ListByConversation(context.Background(), convID, 8203)
	require.NoError(t, err)
	require.NotNil(t, items)
	require.Empty(t, items)
}

// TestFeedbackListByConversationRejectsForeignUser 归属校验与单条反馈一致：非本人会话报会话不存在。
func TestFeedbackListByConversationRejectsForeignUser(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc, _, convID := newFeedbackListFixture(t, 8204, db)

	items, err := svc.ListByConversation(context.Background(), convID, 8205)
	require.Nil(t, items)
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok)
	require.Equal(t, common.RESOURCE_NOT_FOUND.Code, bizErr.Code().Code)
}

// TestFeedbackListByConversationDeletedConversation 会话已软删时不可见。
func TestFeedbackListByConversationDeletedConversation(t *testing.T) {
	db := testutil.NewTestDB(t)
	ctx := context.Background()
	svc, convRepo, convID := newFeedbackListFixture(t, 8206, db)
	require.NoError(t, convRepo.SoftDeleteByIDs(ctx, []int64{convID}, 8206))

	_, err := svc.ListByConversation(ctx, convID, 8206)
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok)
	require.Equal(t, common.RESOURCE_NOT_FOUND.Code, bizErr.Code().Code)
}

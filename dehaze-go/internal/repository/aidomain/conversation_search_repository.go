package aidomain

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
)

// PaginateWithKeyword 关键词会话分页：标题命中或消息内容命中。
//
// 与 python 的差异：python 用户视角 keyword 走 ES 全文检索，Go 无 ES 客户端，
// 用"标题 LIKE + 消息内容 LIKE"等价替代（结果集与 matchedMessageId 语义一致，
// 排序仍遵循置顶优先规则）。
func (r *ConversationRepository) PaginateWithKeyword(ctx context.Context, userID int64, page, size int, keyword string, status *int, admin bool) ([]model.SysAiConversation, int64, error) {
	pattern := "%" + escapeLike(keyword) + "%"
	db := r.db.WithContext(ctx).Model(&model.SysAiConversation{}).Where("deleted = 0")
	if !admin {
		db = db.Where("user_id = ?", userID)
	}
	db = db.Where(
		"title LIKE ? ESCAPE '\\\\' OR EXISTS (SELECT 1 FROM sys_ai_message m WHERE m.conversation_id = sys_ai_conversation.id AND m.deleted = 0 AND m.content LIKE ? ESCAPE '\\\\')",
		pattern, pattern)
	if status != nil {
		db = db.Where("status = ?", *status)
	}
	db = db.Order("pinned DESC, pinned_at DESC, last_message_at DESC, id DESC")
	return countAndFind[model.SysAiConversation](db, page, size)
}

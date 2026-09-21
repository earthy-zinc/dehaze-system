package aidomain

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// MessageRepository AI 消息数据访问。
type MessageRepository struct {
	db *gorm.DB
}

func NewMessageRepository(db *gorm.DB) *MessageRepository {
	return &MessageRepository{db: db}
}

func (r *MessageRepository) GetByID(ctx context.Context, id int64) (*model.SysAiMessage, error) {
	var msg model.SysAiMessage
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&msg).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &msg, err
}

func (r *MessageRepository) GetByIDAndUser(ctx context.Context, id, userID int64) (*model.SysAiMessage, error) {
	var msg model.SysAiMessage
	err := r.db.WithContext(ctx).Raw(
		"SELECT m.* FROM sys_ai_message m JOIN sys_ai_conversation c ON c.id = m.conversation_id "+
			"WHERE m.id = ? AND m.deleted = 0 AND c.user_id = ? AND c.deleted = 0", id, userID).
		Scan(&msg).Error
	if err != nil {
		return nil, err
	}
	if msg.ID == 0 {
		return nil, nil
	}
	return &msg, nil
}

// ListByConversationCursor 会话消息游标分页（id 倒序等价时间倒序）：
//   - 返回 total＝该会话消息总数（不受 before 影响，供前端展示总数）；
//   - before 非空时只取 id < before，否则取最新一页；
//   - 多取一条（limit+1）供调用方判定 hasMore，无需额外一次存在性往返。
func (r *MessageRepository) ListByConversationCursor(ctx context.Context, convID int64, before *int64, limit int) ([]model.SysAiMessage, int64, error) {
	base := r.db.WithContext(ctx).Model(&model.SysAiMessage{}).
		Where("conversation_id = ? AND deleted = 0", convID)
	var total int64
	if err := base.Session(&gorm.Session{}).Count(&total).Error; err != nil {
		return nil, 0, err
	}
	query := base.Order("id DESC")
	if before != nil {
		query = query.Where("id < ?", *before)
	}
	var messages []model.SysAiMessage
	if err := query.Limit(limit + 1).Find(&messages).Error; err != nil {
		return nil, 0, err
	}
	return messages, total, nil
}

// GetChildren 某消息的子消息（分支列表，时间倒序）。
func (r *MessageRepository) GetChildren(ctx context.Context, convID, parentID int64) ([]model.SysAiMessage, error) {
	var msgs []model.SysAiMessage
	err := r.db.WithContext(ctx).
		Where("conversation_id = ? AND parent_message_id = ? AND deleted = 0", convID, parentID).
		Order("id DESC").Find(&msgs).Error
	return msgs, err
}

// GetChainByID 沿 parent 链回溯到根（按 id 正序返回，供会话导出）。
func (r *MessageRepository) GetChainByID(ctx context.Context, convID, tailID int64) ([]model.SysAiMessage, error) {
	var all []model.SysAiMessage
	if err := r.db.WithContext(ctx).
		Where("conversation_id = ? AND deleted = 0", convID).
		Order("id ASC").Find(&all).Error; err != nil {
		return nil, err
	}
	byID := make(map[int64]model.SysAiMessage, len(all))
	for _, m := range all {
		byID[m.ID] = m
	}
	var chain []model.SysAiMessage
	visited := make(map[int64]struct{})
	for cur := tailID; cur != 0; {
		if _, seen := visited[cur]; seen {
			break
		}
		visited[cur] = struct{}{}
		msg, ok := byID[cur]
		if !ok {
			break
		}
		chain = append(chain, msg)
		if msg.ParentMessageID == nil {
			break
		}
		cur = *msg.ParentMessageID
	}
	for i, j := 0, len(chain)-1; i < j; i, j = i+1, j-1 {
		chain[i], chain[j] = chain[j], chain[i]
	}
	return chain, nil
}

func (r *MessageRepository) CountMessagesAfter(ctx context.Context, convID, msgID int64) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysAiMessage{}).
		Where("conversation_id = ? AND deleted = 0 AND id > ?", convID, msgID).
		Count(&count).Error
	return count, err
}

func (r *MessageRepository) GetLastMessageID(ctx context.Context, convID int64) (*int64, error) {
	var row struct {
		MaxID *int64 `gorm:"column:max_id"`
	}
	err := r.db.WithContext(ctx).Model(&model.SysAiMessage{}).
		Select("MAX(id) AS max_id").
		Where("conversation_id = ? AND deleted = 0", convID).
		Scan(&row).Error
	return row.MaxID, err
}

func (r *MessageRepository) SoftDeleteByIDs(ctx context.Context, ids []int64, updateBy int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Model(&model.SysAiMessage{}).
		Where("id IN ? AND deleted = 0", ids).
		Updates(map[string]any{"deleted": gorm.Expr("id"), "update_by": updateBy}).Error
}

// ListAnomalyStatusByConversations 按会话汇总异常消息状态（3:失败;4:已取消）。
func (r *MessageRepository) ListAnomalyStatusByConversations(ctx context.Context, convIDs []int64) (map[int64]map[int]struct{}, error) {
	result := make(map[int64]map[int]struct{})
	if len(convIDs) == 0 {
		return result, nil
	}
	type row struct {
		ConversationID int64 `gorm:"column:conversation_id"`
		Status         int   `gorm:"column:status"`
	}
	var rows []row
	err := r.db.WithContext(ctx).Table("sys_ai_message").
		Select("conversation_id, status").
		Where("conversation_id IN ? AND status IN ? AND deleted = 0", convIDs, []int{3, 4}).
		Group("conversation_id, status").
		Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	for _, item := range rows {
		if result[item.ConversationID] == nil {
			result[item.ConversationID] = make(map[int]struct{})
		}
		result[item.ConversationID][item.Status] = struct{}{}
	}
	return result, nil
}

// ThoughtRepository 推理步骤数据访问。
type ThoughtRepository struct {
	db *gorm.DB
}

func NewThoughtRepository(db *gorm.DB) *ThoughtRepository {
	return &ThoughtRepository{db: db}
}

func (r *ThoughtRepository) ListByMessage(ctx context.Context, messageID int64) ([]model.SysAiAgentThought, error) {
	var items []model.SysAiAgentThought
	err := r.db.WithContext(ctx).
		Where("message_id = ?", messageID).
		Order("position ASC").Find(&items).Error
	return items, err
}

// ListByMessages 批量取多条消息的推理步骤（position 正序），避免逐消息 N+1。
func (r *ThoughtRepository) ListByMessages(ctx context.Context, messageIDs []int64) (map[int64][]model.SysAiAgentThought, error) {
	result := make(map[int64][]model.SysAiAgentThought)
	if len(messageIDs) == 0 {
		return result, nil
	}
	var items []model.SysAiAgentThought
	err := r.db.WithContext(ctx).
		Where("message_id IN ?", messageIDs).
		Order("message_id ASC, position ASC").Find(&items).Error
	if err != nil {
		return nil, err
	}
	for _, item := range items {
		result[item.MessageID] = append(result[item.MessageID], item)
	}
	return result, nil
}

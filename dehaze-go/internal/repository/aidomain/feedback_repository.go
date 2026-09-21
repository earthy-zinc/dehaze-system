package aidomain

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// MessageFeedbackRepository 消息反馈数据访问。
type MessageFeedbackRepository struct {
	db *gorm.DB
}

func NewMessageFeedbackRepository(db *gorm.DB) *MessageFeedbackRepository {
	return &MessageFeedbackRepository{db: db}
}

// GetByUserAndMessage 取未删除反馈。
func (r *MessageFeedbackRepository) GetByUserAndMessage(ctx context.Context, messageID, userID int64) (*model.SysAiMessageFeedback, error) {
	var fb model.SysAiMessageFeedback
	err := r.db.WithContext(ctx).
		Where("message_id = ? AND user_id = ? AND deleted = 0", messageID, userID).
		First(&fb).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &fb, err
}

// Upsert 存在则更新并复活（deleted=0），不存在则插入；查全表以命中唯一索引。
func (r *MessageFeedbackRepository) Upsert(ctx context.Context, fb *model.SysAiMessageFeedback) (*model.SysAiMessageFeedback, error) {
	var existing model.SysAiMessageFeedback
	err := r.db.WithContext(ctx).
		Where("message_id = ? AND user_id = ?", fb.MessageID, fb.UserID).
		First(&existing).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		if err := r.db.WithContext(ctx).Create(fb).Error; err != nil {
			return nil, err
		}
		return fb, nil
	}
	if err != nil {
		return nil, err
	}
	existing.Rating = fb.Rating
	existing.Tags = fb.Tags
	existing.Comment = fb.Comment
	existing.ConversationID = fb.ConversationID
	existing.Model = fb.Model
	existing.Source = fb.Source
	existing.Processed = 0
	existing.ProcessTime = nil
	if err := r.db.WithContext(ctx).Model(&model.SysAiMessageFeedback{}).
		Where("id = ?", existing.ID).
		Updates(map[string]any{
			"rating":          existing.Rating,
			"tags":            existing.Tags,
			"comment":         existing.Comment,
			"conversation_id": existing.ConversationID,
			"model":           existing.Model,
			"source":          existing.Source,
			"processed":       0,
			"process_time":    nil,
			"deleted":         0,
		}).Error; err != nil {
		return nil, err
	}
	existing.Deleted = 0
	existing.Processed = 0
	existing.ProcessTime = nil
	return &existing, nil
}

func (r *MessageFeedbackRepository) SoftDelete(ctx context.Context, messageID, userID int64) error {
	return r.db.WithContext(ctx).Model(&model.SysAiMessageFeedback{}).
		Where("message_id = ? AND user_id = ? AND deleted = 0", messageID, userID).
		Updates(map[string]any{"deleted": gorm.Expr("id"), "update_time": time.Now()}).Error
}

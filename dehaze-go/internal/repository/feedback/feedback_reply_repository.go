package feedback

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

type FeedbackReplyRepository struct {
	db *gorm.DB
}

func NewFeedbackReplyRepository(db *gorm.DB) *FeedbackReplyRepository {
	return &FeedbackReplyRepository{db: db}
}

func (r *FeedbackReplyRepository) Create(ctx context.Context, reply *model.SysFeedbackReply) error {
	return r.db.WithContext(ctx).Create(reply).Error
}

func (r *FeedbackReplyRepository) FindByFeedbackID(ctx context.Context, feedbackID int64) ([]model.SysFeedbackReply, error) {
	var list []model.SysFeedbackReply
	err := r.db.WithContext(ctx).
		Where("feedback_id = ?", feedbackID).
		Order("create_time ASC").
		Find(&list).Error
	return list, err
}

func (r *FeedbackReplyRepository) FindFirstAdminReply(ctx context.Context, feedbackID int64) (*model.SysFeedbackReply, error) {
	var reply model.SysFeedbackReply
	err := r.db.WithContext(ctx).
		Where("feedback_id = ? AND replier_type = 2", feedbackID).
		Order("create_time ASC").
		First(&reply).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &reply, err
}

var _ IFeedbackReplyRepository = (*FeedbackReplyRepository)(nil)

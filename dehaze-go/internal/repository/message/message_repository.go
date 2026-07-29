package message

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"gorm.io/gorm"
)

type MessageRepository struct {
	db *gorm.DB
}

func NewMessageRepository(db *gorm.DB) *MessageRepository {
	return &MessageRepository{db: db}
}

func (r *MessageRepository) FindByID(ctx context.Context, id int64) (*model.SysMessage, error) {
	var msg model.SysMessage
	err := r.db.WithContext(ctx).Where("id = ?", id).First(&msg).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &msg, nil
}

func (r *MessageRepository) FindByBizModuleAndBizID(ctx context.Context, bizModule, bizID string) ([]model.SysMessage, error) {
	if bizModule == "" || bizID == "" {
		return nil, nil
	}
	var msgs []model.SysMessage
	err := r.db.WithContext(ctx).
		Where("biz_module = ? AND biz_id = ? AND deleted = 0", bizModule, bizID).
		Find(&msgs).Error
	return msgs, err
}

func (r *MessageRepository) FindPage(ctx context.Context, userID int64, q *query.MessageQuery) ([]model.SysMessage, int64, error) {
	pageNum := q.PageNum
	if pageNum <= 0 {
		pageNum = 1
	}
	pageSize := q.PageSize
	if pageSize <= 0 {
		pageSize = 20
	}

	db := r.db.WithContext(ctx).Model(&model.SysMessage{}).
		Where("recipient_id = ? AND deleted = 0", userID)

	if q.Type != "" {
		db = db.Where("type = ?", q.Type)
	}
	if q.ReadStatus != nil {
		db = db.Where("read_status = ?", *q.ReadStatus)
	}

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var msgs []model.SysMessage
	err := db.Order("read_status ASC, create_time DESC").
		Offset((pageNum - 1) * pageSize).Limit(pageSize).
		Find(&msgs).Error
	return msgs, total, err
}

func (r *MessageRepository) SearchPage(ctx context.Context, userID int64, q *query.MessageSearchQuery) ([]model.SysMessage, int64, error) {
	pageNum := q.PageNum
	if pageNum <= 0 {
		pageNum = 1
	}
	pageSize := q.PageSize
	if pageSize <= 0 {
		pageSize = 20
	}

	keyword := "%" + q.Keyword + "%"
	db := r.db.WithContext(ctx).Model(&model.SysMessage{}).
		Where("recipient_id = ? AND deleted = 0 AND (title LIKE ? OR content LIKE ?)", userID, keyword, keyword)

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var msgs []model.SysMessage
	err := db.Order("read_status ASC, create_time DESC").
		Offset((pageNum - 1) * pageSize).Limit(pageSize).
		Find(&msgs).Error
	return msgs, total, err
}

func (r *MessageRepository) CountUnread(ctx context.Context, userID int64) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysMessage{}).
		Where("recipient_id = ? AND read_status = 0 AND deleted = 0", userID).
		Count(&count).Error
	return count, err
}

func (r *MessageRepository) Create(ctx context.Context, msg *model.SysMessage) error {
	return r.db.WithContext(ctx).Create(msg).Error
}

func (r *MessageRepository) CreateBatch(ctx context.Context, msgs []model.SysMessage) ([]int64, error) {
	if len(msgs) == 0 {
		return nil, nil
	}
	if err := r.db.WithContext(ctx).Create(&msgs).Error; err != nil {
		return nil, err
	}
	ids := make([]int64, 0, len(msgs))
	for i := range msgs {
		ids = append(ids, msgs[i].ID)
	}
	return ids, nil
}

func (r *MessageRepository) MarkRead(ctx context.Context, id, userID int64) (int64, error) {
	result := r.db.WithContext(ctx).Model(&model.SysMessage{}).
		Where("id = ? AND recipient_id = ? AND read_status = 0 AND deleted = 0", id, userID).
		Updates(map[string]interface{}{
			"read_status": 1,
			"read_time":   time.Now(),
			"update_by":   userID,
		})
	return result.RowsAffected, result.Error
}

func (r *MessageRepository) MarkAllRead(ctx context.Context, userID int64, msgType string) (int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysMessage{}).
		Where("recipient_id = ? AND read_status = 0 AND deleted = 0", userID)
	if msgType != "" {
		db = db.Where("type = ?", msgType)
	}
	result := db.Updates(map[string]interface{}{
		"read_status": 1,
		"read_time":   time.Now(),
		"update_by":   userID,
	})
	return result.RowsAffected, result.Error
}

func (r *MessageRepository) SoftDelete(ctx context.Context, ids []int64, userID int64) error {
	return r.db.WithContext(ctx).Model(&model.SysMessage{}).
		Where("id IN ? AND recipient_id = ?", ids, userID).
		Update("deleted", 1).Error
}

func (r *MessageRepository) DeleteExpiredBatch(ctx context.Context, before time.Time, batchSize int) (int64, error) {
	var total int64
	for {
		result := r.db.WithContext(ctx).
			Where("expires_at < ?", before).
			Limit(batchSize).
			Delete(&model.SysMessage{})
		if result.Error != nil {
			return total, result.Error
		}
		total += result.RowsAffected
		if result.RowsAffected < int64(batchSize) {
			break
		}
	}
	return total, nil
}

var _ IMessageRepository = (*MessageRepository)(nil)

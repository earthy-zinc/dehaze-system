package message

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"gorm.io/gorm"
)

type MessageTemplateRepository struct {
	db *gorm.DB
}

func NewMessageTemplateRepository(db *gorm.DB) *MessageTemplateRepository {
	return &MessageTemplateRepository{db: db}
}

func (r *MessageTemplateRepository) FindByID(ctx context.Context, id int64) (*model.SysMessageTemplate, error) {
	var tpl model.SysMessageTemplate
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&tpl).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &tpl, nil
}

func (r *MessageTemplateRepository) FindByCode(ctx context.Context, code string) (*model.SysMessageTemplate, error) {
	var tpl model.SysMessageTemplate
	err := r.db.WithContext(ctx).Where("code = ? AND deleted = 0", code).First(&tpl).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &tpl, nil
}

func (r *MessageTemplateRepository) FindPage(ctx context.Context, q *query.MessageTemplateQuery) ([]model.SysMessageTemplate, int64, error) {
	pageNum := q.PageNum
	if pageNum <= 0 {
		pageNum = 1
	}
	pageSize := q.PageSize
	if pageSize <= 0 {
		pageSize = 20
	}

	db := r.db.WithContext(ctx).Model(&model.SysMessageTemplate{}).Where("deleted = 0")

	if q.Name != "" {
		db = db.Where("name LIKE ?", "%"+q.Name+"%")
	}
	if q.Type != "" {
		db = db.Where("type = ?", q.Type)
	}
	if q.Status > 0 {
		db = db.Where("status = ?", q.Status)
	}

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var tpls []model.SysMessageTemplate
	err := db.Order("create_time DESC").
		Offset((pageNum - 1) * pageSize).Limit(pageSize).
		Find(&tpls).Error
	return tpls, total, err
}

func (r *MessageTemplateRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).Model(&model.SysMessageTemplate{}).
		Where("id = ? AND deleted = 0", id).
		Updates(updates).Error
}

var _ IMessageTemplateRepository = (*MessageTemplateRepository)(nil)

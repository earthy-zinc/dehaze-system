package message

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"gorm.io/gorm"
)

type AnnouncementRepository struct {
	db *gorm.DB
}

func NewAnnouncementRepository(db *gorm.DB) *AnnouncementRepository {
	return &AnnouncementRepository{db: db}
}

func (r *AnnouncementRepository) FindByID(ctx context.Context, id int64) (*model.SysAnnouncement, error) {
	var ann model.SysAnnouncement
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&ann).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &ann, nil
}

func (r *AnnouncementRepository) FindPendingScheduled(ctx context.Context, before time.Time) ([]model.SysAnnouncement, error) {
	var list []model.SysAnnouncement
	err := r.db.WithContext(ctx).
		Where("status = 2 AND send_time <= ? AND deleted = 0", before).
		Find(&list).Error
	return list, err
}

func (r *AnnouncementRepository) FindPage(ctx context.Context, q *query.AnnouncementQuery) ([]model.SysAnnouncement, int64, error) {
	pageNum := q.PageNum
	if pageNum <= 0 {
		pageNum = 1
	}
	pageSize := q.PageSize
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).Model(&model.SysAnnouncement{}).Where("deleted = 0")

	if q.Title != "" {
		db = db.Where("title LIKE ?", "%"+q.Title+"%")
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

	var anns []model.SysAnnouncement
	err := db.Order("create_time DESC").
		Offset((pageNum - 1) * pageSize).Limit(pageSize).
		Find(&anns).Error
	return anns, total, err
}

func (r *AnnouncementRepository) Create(ctx context.Context, ann *model.SysAnnouncement) (int64, error) {
	if err := r.db.WithContext(ctx).Create(ann).Error; err != nil {
		return 0, err
	}
	return ann.ID, nil
}

func (r *AnnouncementRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).Model(&model.SysAnnouncement{}).
		Where("id = ? AND deleted = 0", id).
		Updates(updates).Error
}

func (r *AnnouncementRepository) SoftDelete(ctx context.Context, id int64) error {
	return r.db.WithContext(ctx).Model(&model.SysAnnouncement{}).
		Where("id = ?", id).
		Update("deleted", 1).Error
}

var _ IAnnouncementRepository = (*AnnouncementRepository)(nil)

package pkgsale

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"gorm.io/gorm"
)

type PromotionRepository struct {
	db *gorm.DB
}

func NewPromotionRepository(db *gorm.DB) *PromotionRepository {
	return &PromotionRepository{db: db}
}

func (r *PromotionRepository) FindByID(ctx context.Context, id int64) (*model.SysPromotion, error) {
	var p model.SysPromotion
	err := r.db.WithContext(ctx).
		Where("id = ? AND deleted = 0", id).
		First(&p).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &p, err
}

func (r *PromotionRepository) FindPage(ctx context.Context, q *query.PromotionPageQuery) ([]model.SysPromotion, int64, error) {
	pageNum := q.PageNum
	if pageNum <= 0 {
		pageNum = 1
	}
	pageSize := q.PageSize
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).Model(&model.SysPromotion{}).Where("deleted = 0")
	if q.Name != "" {
		db = db.Where("name LIKE ?", "%"+q.Name+"%")
	}
	if q.Type != "" {
		db = db.Where("type = ?", q.Type)
	}
	if q.Status != nil {
		db = db.Where("status = ?", *q.Status)
	}
	if q.StartTime != "" {
		db = db.Where("start_time >= ?", q.StartTime)
	}
	if q.EndTime != "" {
		db = db.Where("end_time <= ?", q.EndTime)
	}

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var list []model.SysPromotion
	err := db.Order("id DESC").
		Offset((pageNum - 1) * pageSize).Limit(pageSize).
		Find(&list).Error
	return list, total, err
}

func (r *PromotionRepository) Create(ctx context.Context, p *model.SysPromotion) error {
	return r.db.WithContext(ctx).Create(p).Error
}

func (r *PromotionRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysPromotion{}).
		Where("id = ? AND deleted = 0", id).
		Updates(updates).Error
}

func (r *PromotionRepository) UpdateStatus(ctx context.Context, id int64, status int8) error {
	return r.db.WithContext(ctx).
		Model(&model.SysPromotion{}).
		Where("id = ? AND deleted = 0", id).
		Update("status", status).Error
}

func (r *PromotionRepository) DeleteByID(ctx context.Context, id int64) error {
	return r.db.WithContext(ctx).
		Model(&model.SysPromotion{}).
		Where("id = ? AND deleted = 0", id).
		Update("deleted", gorm.Expr("id")).Error
}

func (r *PromotionRepository) ListPackageIDs(ctx context.Context, promotionID int64) ([]int64, error) {
	var ids []int64
	err := r.db.WithContext(ctx).
		Model(&model.SysPromotionPackage{}).
		Where("promotion_id = ?", promotionID).
		Pluck("package_id", &ids).Error
	return ids, err
}

// RebindPackages 事务内重建活动-套餐关联，折扣方式/折扣值随活动规则统一写入
func (r *PromotionRepository) RebindPackages(ctx context.Context, promotionID int64, discountType string, discountValue int64, packageIDs []int64) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		if err := tx.WithContext(ctx).
			Where("promotion_id = ?", promotionID).
			Delete(&model.SysPromotionPackage{}).Error; err != nil {
			return err
		}
		if len(packageIDs) == 0 {
			return nil
		}
		rows := make([]model.SysPromotionPackage, 0, len(packageIDs))
		for _, packageID := range packageIDs {
			rows = append(rows, model.SysPromotionPackage{
				PromotionID:   promotionID,
				PackageID:     packageID,
				DiscountType:  discountType,
				DiscountValue: discountValue,
			})
		}
		return tx.WithContext(ctx).Create(&rows).Error
	})
}

var _ IPromotionRepository = (*PromotionRepository)(nil)

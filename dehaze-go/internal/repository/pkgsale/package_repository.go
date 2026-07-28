package pkgsale

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"gorm.io/gorm"
)

type PackageRepository struct {
	db *gorm.DB
}

func NewPackageRepository(db *gorm.DB) *PackageRepository {
	return &PackageRepository{db: db}
}

func (r *PackageRepository) FindByID(ctx context.Context, id int64) (*model.SysPackage, error) {
	var p model.SysPackage
	err := r.db.WithContext(ctx).
		Where("id = ? AND deleted = 0", id).
		First(&p).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &p, err
}

func (r *PackageRepository) FindByIDs(ctx context.Context, ids []int64) ([]model.SysPackage, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	var list []model.SysPackage
	err := r.db.WithContext(ctx).
		Where("id IN ? AND deleted = 0", ids).
		Find(&list).Error
	return list, err
}

func (r *PackageRepository) FindAllOnSale(ctx context.Context) ([]model.SysPackage, error) {
	var list []model.SysPackage
	err := r.db.WithContext(ctx).
		Where("status = 1 AND deleted = 0").
		Order("sort ASC, id ASC").
		Find(&list).Error
	return list, err
}

func (r *PackageRepository) FindPage(ctx context.Context, q *query.PackagePageQuery) ([]model.SysPackage, int64, error) {
	pageNum := q.PageNum
	if pageNum <= 0 {
		pageNum = 1
	}
	pageSize := q.PageSize
	if pageSize <= 0 {
		pageSize = 10
	}

	db := r.db.WithContext(ctx).Model(&model.SysPackage{}).Where("deleted = 0")
	if q.Name != "" {
		db = db.Where("name LIKE ?", "%"+q.Name+"%")
	}
	if q.LevelCode != "" {
		db = db.Where("level_code = ?", q.LevelCode)
	}
	if q.Period != "" {
		db = db.Where("period = ?", q.Period)
	}
	if q.Status != nil {
		db = db.Where("status = ?", *q.Status)
	}
	if q.StartTime != "" {
		db = db.Where("create_time >= ?", q.StartTime)
	}
	if q.EndTime != "" {
		db = db.Where("create_time <= ?", q.EndTime)
	}

	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var list []model.SysPackage
	err := db.Order("sort ASC, id DESC").
		Offset((pageNum - 1) * pageSize).Limit(pageSize).
		Find(&list).Error
	return list, total, err
}

func (r *PackageRepository) Create(ctx context.Context, p *model.SysPackage) error {
	return r.db.WithContext(ctx).Create(p).Error
}

func (r *PackageRepository) Update(ctx context.Context, id int64, updates map[string]interface{}) error {
	return r.db.WithContext(ctx).
		Model(&model.SysPackage{}).
		Where("id = ? AND deleted = 0", id).
		Updates(updates).Error
}

func (r *PackageRepository) UpdateStatus(ctx context.Context, id int64, status int8) error {
	return r.db.WithContext(ctx).
		Model(&model.SysPackage{}).
		Where("id = ? AND deleted = 0", id).
		Update("status", status).Error
}

func (r *PackageRepository) DeleteByIDs(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).
		Model(&model.SysPackage{}).
		Where("id IN ? AND deleted = 0", ids).
		Update("deleted", 1).Error
}

func (r *PackageRepository) IncrementSalesCount(ctx context.Context, id int64, delta int64) error {
	return r.db.WithContext(ctx).
		Model(&model.SysPackage{}).
		Where("id = ? AND deleted = 0", id).
		UpdateColumn("sales_count", gorm.Expr("sales_count + ?", delta)).Error
}

func (r *PackageRepository) CountOrders(ctx context.Context, packageID int64) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Table("sys_order").
		Where("package_id = ? AND deleted = 0", packageID).
		Count(&count).Error
	return count, err
}

var _ IPackageRepository = (*PackageRepository)(nil)

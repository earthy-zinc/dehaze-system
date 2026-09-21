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
	if q.PackageType != "" {
		db = db.Where("package_type = ?", q.PackageType)
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
		Update("deleted", gorm.Expr("id")).Error
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

func (r *PackageRepository) CountPaidOrdersByUser(ctx context.Context, userID int64) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Table("sys_order").
		Where("user_id = ? AND status IN ? AND deleted = 0", userID, []int8{2, 3}).
		Count(&count).Error
	return count, err
}

// ExistsByName 检查包名是否存在（仅活跃行，软删行不占唯一键位可重建）
func (r *PackageRepository) ExistsByName(ctx context.Context, name string, excludeID ...int64) (bool, error) {
	var count int64
	query := r.db.WithContext(ctx).Model(&model.SysPackage{}).
		Where("name = ?", name)
	if len(excludeID) > 0 {
		query = query.Where("id != ?", excludeID[0])
	}
	err := query.Count(&count).Error
	return count > 0, err
}

func (r *PackageRepository) FindActivePromotionsByPackageID(ctx context.Context, packageID int64) ([]PromotionWithPackage, error) {
	var rows []PromotionWithPackage
	err := r.db.WithContext(ctx).
		Table("sys_promotion_package pp").
		Select("pp.discount_type, pp.discount_value, p.id AS promotion_id, p.name, p.type, p.description, p.status, p.start_time, p.end_time, p.activity_rules, p.new_user_only").
		Joins("JOIN sys_promotion p ON pp.promotion_id = p.id").
		Where("pp.package_id = ? AND p.deleted = 0", packageID).
		Scan(&rows).Error
	return rows, err
}

func (r *PackageRepository) SumPaidAmountByStatus(ctx context.Context, statuses []int8) (int64, error) {
	var total int64
	err := r.db.WithContext(ctx).
		Table("sys_order").
		Where("status IN ? AND deleted = 0", statuses).
		Select("COALESCE(SUM(paid_amount), 0)").
		Scan(&total).Error
	return total, err
}

func (r *PackageRepository) CountOrdersByStatus(ctx context.Context, statuses []int8) (int64, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Table("sys_order").
		Where("status IN ? AND deleted = 0", statuses).
		Count(&count).Error
	return count, err
}

func (r *PackageRepository) GetPackageOrderStats(ctx context.Context, statuses []int8) ([]PackageOrderStatRow, error) {
	var rows []PackageOrderStatRow
	err := r.db.WithContext(ctx).
		Table("sys_order").
		Select("package_id, package_name, COUNT(*) as count, COALESCE(SUM(paid_amount), 0) as revenue").
		Where("status IN ? AND deleted = 0", statuses).
		Group("package_id, package_name").
		Scan(&rows).Error
	return rows, err
}

func (r *PackageRepository) GetLevelOrderStats(ctx context.Context, statuses []int8) ([]LevelOrderStatRow, error) {
	var rows []LevelOrderStatRow
	err := r.db.WithContext(ctx).
		Table("sys_order").
		Select("package_level, COUNT(*) as count, COALESCE(SUM(paid_amount), 0) as revenue").
		Where("status IN ? AND deleted = 0", statuses).
		Group("package_level").
		Scan(&rows).Error
	return rows, err
}

func (r *PackageRepository) GetPeriodOrderStats(ctx context.Context, statuses []int8) ([]PeriodOrderStatRow, error) {
	var rows []PeriodOrderStatRow
	err := r.db.WithContext(ctx).
		Table("sys_order o").
		Select("p.period, COUNT(*) as count, COALESCE(SUM(o.paid_amount), 0) as revenue").
		Joins("JOIN sys_package p ON o.package_id = p.id").
		Where("o.status IN ? AND o.deleted = 0 AND p.deleted = 0", statuses).
		Group("p.period").
		Scan(&rows).Error
	return rows, err
}

// FindActiveVipByLevelCode 按关联等级查询在售 VIP 套餐（排序最前的一条）。
// 对齐 python `package_repository.get_by_level_code`：未删除、status=1、package_type='vip'、
// level_code 匹配，按 sort asc, id asc 取第一条；无匹配返回 nil。
func (r *PackageRepository) FindActiveVipByLevelCode(ctx context.Context, levelCode string) (*model.SysPackage, error) {
	var p model.SysPackage
	err := r.db.WithContext(ctx).
		Where("deleted = 0 AND status = 1 AND package_type = ? AND level_code = ?", "vip", levelCode).
		Order("sort ASC, id ASC").
		First(&p).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &p, nil
}

var _ IPackageRepository = (*PackageRepository)(nil)

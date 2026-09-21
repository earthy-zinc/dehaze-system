package ai

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// 注：LIKE 转义复用同包 model_repository.go 的 escapeLike。

// CostRepository 模型成本单价（版本 + 档位明细，方案 A 软删）与毛利核算数据源。
type CostRepository struct {
	db *gorm.DB
}

func NewCostRepository(db *gorm.DB) *CostRepository {
	return &CostRepository{db: db}
}

// NextPriceVersion 同模型同供应商的价格版本号递增（含软删历史，版本号不可复用）
func (r *CostRepository) NextPriceVersion(ctx context.Context, modelID string, providerID int64) (int, error) {
	var maxVersion *int
	err := r.db.WithContext(ctx).Model(&model.SysAiModelCost{}).
		Select("MAX(price_version)").
		Where("model_id = ? AND provider_id = ?", modelID, providerID).
		Scan(&maxVersion).Error
	if err != nil {
		return 0, err
	}
	if maxVersion == nil {
		return 1, nil
	}
	return *maxVersion + 1, nil
}

// CreateCost 新增成本价格版本
func (r *CostRepository) CreateCost(ctx context.Context, cost *model.SysAiModelCost) error {
	return r.db.WithContext(ctx).Create(cost).Error
}

// CreateCostDetails 批量新增档位明细
func (r *CostRepository) CreateCostDetails(ctx context.Context, details []model.SysAiModelCostDetail) error {
	if len(details) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Create(&details).Error
}

// GetCost 按主键查询未软删的成本版本
func (r *CostRepository) GetCost(ctx context.Context, id int64) (*model.SysAiModelCost, error) {
	var item model.SysAiModelCost
	err := r.db.WithContext(ctx).Where("id = ? AND deleted = 0", id).First(&item).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &item, err
}

// UpdateCost 更新成本版本主表字段
func (r *CostRepository) UpdateCost(ctx context.Context, id int64, fields map[string]any) error {
	fields["update_time"] = time.Now()
	return r.db.WithContext(ctx).Model(&model.SysAiModelCost{}).Where("id = ?", id).Updates(fields).Error
}

// ListCostDetails 查询价格版本的档位明细（id 正序）
func (r *CostRepository) ListCostDetails(ctx context.Context, priceID int64) ([]model.SysAiModelCostDetail, error) {
	var items []model.SysAiModelCostDetail
	err := r.db.WithContext(ctx).
		Where("price_id = ? AND deleted = 0", priceID).Order("id ASC").Find(&items).Error
	return items, err
}

// PaginateCosts 成本单价分页（create_time/id 倒序，keyword 匹配 model_id）
func (r *CostRepository) PaginateCosts(
	ctx context.Context, page, size int, keyword, modelID string, providerID *int64,
) ([]model.SysAiModelCost, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiModelCost{}).Where("deleted = 0")
	if modelID != "" {
		db = db.Where("model_id = ?", modelID)
	}
	if providerID != nil {
		db = db.Where("provider_id = ?", *providerID)
	}
	if keyword != "" {
		pattern := "%" + escapeLike(keyword) + "%"
		db = db.Where("model_id LIKE ? ESCAPE '\\\\'", pattern)
	}
	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	var items []model.SysAiModelCost
	err := db.Order("create_time DESC, id DESC").Offset((page - 1) * size).Limit(size).Find(&items).Error
	return items, total, err
}

// SoftDeleteCost 逻辑删除成本版本（deleted = id，方案 A）
func (r *CostRepository) SoftDeleteCost(ctx context.Context, id, updateBy int64) error {
	return r.db.WithContext(ctx).Model(&model.SysAiModelCost{}).Where("id = ?", id).
		Updates(map[string]any{
			"deleted":     gorm.Expr("id"),
			"update_time": time.Now(),
			"update_by":   updateBy,
		}).Error
}

// SoftDeleteCostDetails 逻辑删除价格版本下的全部档位明细
func (r *CostRepository) SoftDeleteCostDetails(ctx context.Context, priceID, updateBy int64) error {
	return r.db.WithContext(ctx).Model(&model.SysAiModelCostDetail{}).Where("price_id = ? AND deleted = 0", priceID).
		Updates(map[string]any{
			"deleted":     gorm.Expr("id"),
			"update_time": time.Now(),
			"update_by":   updateBy,
		}).Error
}

// SumBillingCost 计费记录成本合计（元）
func (r *CostRepository) SumBillingCost(ctx context.Context, start, end *time.Time) (float64, error) {
	var cost float64
	db := r.db.WithContext(ctx).Model(&model.SysAiBilling{}).Select("COALESCE(SUM(cost), 0)")
	if start != nil {
		db = db.Where("create_time >= ?", *start)
	}
	if end != nil {
		db = db.Where("create_time <= ?", *end)
	}
	err := db.Scan(&cost).Error
	return cost, err
}

// SumPaidOrderAmountByType 已实收订单金额（分）按商品类型汇总（status 2=已支付 / 3=已完成）
func (r *CostRepository) SumPaidOrderAmountByType(ctx context.Context, start, end *time.Time) (map[string]int64, error) {
	var rows []struct {
		PackageType string `gorm:"column:package_type"`
		Amount      int64  `gorm:"column:amount"`
	}
	db := r.db.WithContext(ctx).Table("sys_order").
		Select("package_type, COALESCE(SUM(paid_amount), 0) AS amount").
		Where("status IN ?", []int{2, 3})
	if start != nil {
		db = db.Where("paid_time >= ?", *start)
	}
	if end != nil {
		db = db.Where("paid_time <= ?", *end)
	}
	if err := db.Group("package_type").Scan(&rows).Error; err != nil {
		return nil, err
	}
	result := make(map[string]int64, len(rows))
	for _, row := range rows {
		result[row.PackageType] = row.Amount
	}
	return result, nil
}

// CostGroupRow 成本分组行
type CostGroupRow struct {
	Dimension string  `gorm:"column:dimension"`
	Cost      float64 `gorm:"column:cost"`
}

// SumBillingCostByGroup 成本按模型/供应商分组分解（订单实收无法归因，分组行仅含成本）
func (r *CostRepository) SumBillingCostByGroup(
	ctx context.Context, dimension string, start, end *time.Time, modelID string, providerID *int64,
) ([]CostGroupRow, error) {
	dimExpr, err := costDimExpr(dimension)
	if err != nil {
		return nil, err
	}
	var rows []CostGroupRow
	db := r.db.WithContext(ctx).Model(&model.SysAiBilling{}).
		Select(fmt.Sprintf("%s AS dimension, COALESCE(SUM(cost), 0) AS cost", dimExpr)).
		Where("cost IS NOT NULL")
	if start != nil {
		db = db.Where("create_time >= ?", *start)
	}
	if end != nil {
		db = db.Where("create_time <= ?", *end)
	}
	if modelID != "" {
		db = db.Where("model = ?", modelID)
	}
	if providerID != nil {
		db = db.Where("provider_id = ?", *providerID)
	}
	err = db.Group("dimension").Order("dimension ASC").Scan(&rows).Error
	return rows, err
}

func costDimExpr(dimension string) (string, error) {
	switch dimension {
	case "model":
		return "model", nil
	case "provider":
		return "provider_id", nil
	default:
		return "", fmt.Errorf("unsupported cost stat dimension: %s", dimension)
	}
}

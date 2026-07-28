package pkgsale

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	memberrepo "github.com/earthyzinc/dehaze-go/internal/repository/member"
	pkgsalerepo "github.com/earthyzinc/dehaze-go/internal/repository/pkgsale"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"gorm.io/gorm"
)

const (
	timeFormat = "2006-01-02 15:04:05"
)

var levelNames = map[string]string{
	"level_0": "普通用户",
	"level_1": "VIP1",
	"level_2": "VIP2",
	"level_3": "SVIP",
}

var periodNames = map[string]string{
	"monthly":   "月度",
	"quarterly": "季度",
	"yearly":    "年度",
}

type PackageService struct {
	db            *gorm.DB
	packageRepo   pkgsalerepo.IPackageRepository
	couponRepo    pkgsalerepo.ICouponRepository
	userCouponRepo pkgsalerepo.IUserCouponRepository
	benefitRepo   memberrepo.IMemberBenefitRepository
}

func NewPackageService(
	db *gorm.DB,
	packageRepo pkgsalerepo.IPackageRepository,
	couponRepo pkgsalerepo.ICouponRepository,
	userCouponRepo pkgsalerepo.IUserCouponRepository,
	benefitRepo memberrepo.IMemberBenefitRepository,
) *PackageService {
	return &PackageService{
		db:            db,
		packageRepo:   packageRepo,
		couponRepo:    couponRepo,
		userCouponRepo: userCouponRepo,
		benefitRepo:   benefitRepo,
	}
}

func (s *PackageService) ListOnSale(ctx context.Context) ([]vo.PackageDetailVO, error) {
	list, err := s.packageRepo.FindAllOnSale(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询在售套餐失败", err)
	}

	benefits, err := s.benefitRepo.FindAll(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询权益配置失败", err)
	}
	benefitMap := make(map[string]*model.SysMemberBenefit, len(benefits))
	for i := range benefits {
		benefitMap[benefits[i].LevelCode] = &benefits[i]
	}

	result := make([]vo.PackageDetailVO, 0, len(list))
	for _, p := range list {
		result = append(result, s.toPackageDetailVO(&p, benefitMap))
	}
	return result, nil
}

func (s *PackageService) GetDetail(ctx context.Context, id int64) (*vo.PackageDetailVO, error) {
	p, err := s.packageRepo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询套餐失败", err)
	}
	if p == nil {
		return nil, common.NewBizError(common.PACKAGE_NOT_FOUND, "套餐不存在")
	}

	benefits, err := s.benefitRepo.FindAll(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询权益配置失败", err)
	}
	benefitMap := make(map[string]*model.SysMemberBenefit, len(benefits))
	for i := range benefits {
		benefitMap[benefits[i].LevelCode] = &benefits[i]
	}

	vo := s.toPackageDetailVO(p, benefitMap)
	return &vo, nil
}

func (s *PackageService) CalculatePrice(ctx context.Context, userID, packageID int64, userCouponID *int64) (*vo.PriceResult, error) {
	p, err := s.packageRepo.FindByID(ctx, packageID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询套餐失败", err)
	}
	if p == nil {
		return nil, common.NewBizError(common.PACKAGE_NOT_FOUND, "套餐不存在")
	}

	result := &vo.PriceResult{
		OriginalPrice:  p.OriginalPrice,
		DiscountAmount: p.OriginalPrice - p.SalePrice,
		PayableAmount:  p.SalePrice,
	}

	if userCouponID != nil && *userCouponID > 0 {
		uc, err := s.userCouponRepo.FindByUserIDAndStatusForUpdate(ctx, userID, *userCouponID)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询优惠券失败", err)
		}
		if uc == nil {
			return nil, common.NewBizError(common.COUPON_NOT_FOUND, "优惠券不存在")
		}

		c, err := s.couponRepo.FindByID(ctx, uc.CouponID)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询优惠券模板失败", err)
		}
		if c == nil {
			return nil, common.NewBizError(common.COUPON_NOT_FOUND, "优惠券不存在")
		}

		if !isCouponApplicable(c, packageID) {
			return nil, common.NewBizError(common.COUPON_NOT_APPLICABLE, "优惠券不适用于该套餐")
		}

		couponAmount := calcCouponAmount(c, result.PayableAmount)
		if couponAmount > result.PayableAmount {
			couponAmount = result.PayableAmount
		}
		result.CouponAmount = couponAmount
		result.PayableAmount = result.PayableAmount - couponAmount
	}

	return result, nil
}

func (s *PackageService) GetPage(ctx context.Context, q *query.PackagePageQuery) (*vo.PageResult[vo.PackagePageVO], error) {
	list, total, err := s.packageRepo.FindPage(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询套餐列表失败", err)
	}
	vos := make([]vo.PackagePageVO, 0, len(list))
	for _, p := range list {
		vos = append(vos, vo.PackagePageVO{
			ID:            p.ID,
			Name:          p.Name,
			LevelCode:     p.LevelCode,
			LevelName:     getLevelName(p.LevelCode),
			Period:        p.Period,
			PeriodDays:    p.PeriodDays,
			OriginalPrice: p.OriginalPrice,
			SalePrice:     p.SalePrice,
			DailyPrice:    calcDailyPrice(p.SalePrice, p.PeriodDays),
			SalesCount:    p.SalesCount,
			Status:        int(p.Status),
			CreateTime:    p.CreatedAt.Format(timeFormat),
		})
	}
	return &vo.PageResult[vo.PackagePageVO]{List: vos, Total: total}, nil
}

func (s *PackageService) GetForm(ctx context.Context, id int64) (*bo.PackageForm, error) {
	p, err := s.packageRepo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询套餐失败", err)
	}
	if p == nil {
		return nil, common.NewBizError(common.PACKAGE_NOT_FOUND, "套餐不存在")
	}

	form := &bo.PackageForm{
		ID:            p.ID,
		Name:          p.Name,
		LevelCode:     p.LevelCode,
		Period:        p.Period,
		PeriodDays:    p.PeriodDays,
		OriginalPrice: p.OriginalPrice,
		SalePrice:     p.SalePrice,
		Description:   p.Description,
		Sort:          &p.Sort,
		Status:        intPtr(int(p.Status)),
	}

	if p.BenefitOverrides != "" {
		var overrides bo.BenefitOverrides
		if err := json.Unmarshal([]byte(p.BenefitOverrides), &overrides); err == nil {
			form.BenefitOverrides = &overrides
		}
	}

	return form, nil
}

func (s *PackageService) Create(ctx context.Context, form *bo.PackageForm) error {
	if form.Name == "" {
		return common.NewBizError(common.PARAM_ERROR, "套餐名称不能为空")
	}
	if form.OriginalPrice < 0 || form.SalePrice < 0 {
		return common.NewBizError(common.PARAM_ERROR, "价格不能为负数")
	}

	p := &model.SysPackage{
		Name:          form.Name,
		LevelCode:     form.LevelCode,
		Period:        form.Period,
		PeriodDays:    form.PeriodDays,
		OriginalPrice: form.OriginalPrice,
		SalePrice:     form.SalePrice,
		Description:   form.Description,
		Sort:          0,
		Status:        0,
	}

	if form.Sort != nil {
		p.Sort = *form.Sort
	}
	if form.Status != nil {
		p.Status = int8(*form.Status)
	}

	if form.BenefitOverrides != nil {
		data, err := json.Marshal(form.BenefitOverrides)
		if err == nil {
			p.BenefitOverrides = string(data)
		}
	}

	if err := s.packageRepo.Create(ctx, p); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "创建套餐失败", err)
	}
	return nil
}

func (s *PackageService) Update(ctx context.Context, id int64, form *bo.PackageForm) error {
	p, err := s.packageRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询套餐失败", err)
	}
	if p == nil {
		return common.NewBizError(common.PACKAGE_NOT_FOUND, "套餐不存在")
	}

	if form.OriginalPrice < 0 || form.SalePrice < 0 {
		return common.NewBizError(common.PARAM_ERROR, "价格不能为负数")
	}

	updates := map[string]interface{}{
		"name":           form.Name,
		"level_code":     form.LevelCode,
		"period":         form.Period,
		"period_days":    form.PeriodDays,
		"original_price": form.OriginalPrice,
		"sale_price":     form.SalePrice,
		"description":    form.Description,
	}

	if form.Sort != nil {
		updates["sort"] = *form.Sort
	}
	if form.Status != nil {
		updates["status"] = int8(*form.Status)
	}

	if form.BenefitOverrides != nil {
		data, err := json.Marshal(form.BenefitOverrides)
		if err == nil {
			updates["benefit_overrides"] = string(data)
		}
	} else {
		updates["benefit_overrides"] = ""
	}

	if err := s.packageRepo.Update(ctx, id, updates); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新套餐失败", err)
	}
	return nil
}

func (s *PackageService) UpdateStatus(ctx context.Context, id int64, status int) error {
	p, err := s.packageRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询套餐失败", err)
	}
	if p == nil {
		return common.NewBizError(common.PACKAGE_NOT_FOUND, "套餐不存在")
	}

	if err := s.packageRepo.UpdateStatus(ctx, id, int8(status)); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新套餐状态失败", err)
	}
	return nil
}

func (s *PackageService) DeleteByIDs(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return common.NewBizError(common.PARAM_ERROR, "请选择要删除的套餐")
	}

	for _, id := range ids {
		p, err := s.packageRepo.FindByID(ctx, id)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询套餐失败", err)
		}
		if p == nil {
			return common.NewBizError(common.PACKAGE_NOT_FOUND, fmt.Sprintf("套餐(id=%d)不存在", id))
		}

		count, err := s.packageRepo.CountOrders(ctx, id)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询套餐订单数失败", err)
		}
		if count > 0 {
			return common.NewBizError(common.PACKAGE_HAS_ORDERS, fmt.Sprintf("套餐(%s)下已有关联订单，无法删除", p.Name))
		}
	}

	if err := s.packageRepo.DeleteByIDs(ctx, ids); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除套餐失败", err)
	}
	return nil
}

func (s *PackageService) GetSalesStats(ctx context.Context) (*vo.SalesStatsVO, error) {
	list, err := s.packageRepo.FindAllOnSale(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询套餐列表失败", err)
	}

	stats := &vo.SalesStatsVO{
		PackageStats: make([]vo.PackageSalesStatItem, 0),
		LevelStats:   make([]vo.LevelSalesStatItem, 0),
		PeriodStats:  make([]vo.PeriodSalesStatItem, 0),
		CouponStats:  vo.CouponStatsVO{},
	}

	levelAgg := make(map[string]*vo.LevelSalesStatItem)
	periodAgg := make(map[string]*vo.PeriodSalesStatItem)

	for _, p := range list {
		stats.TotalSales += p.SalesCount
		stats.TotalRevenue += p.SalesCount * p.SalePrice

		stats.PackageStats = append(stats.PackageStats, vo.PackageSalesStatItem{
			PackageID:   p.ID,
			PackageName: p.Name,
			SalesCount:  p.SalesCount,
			Revenue:     p.SalesCount * p.SalePrice,
		})

		if item, ok := levelAgg[p.LevelCode]; ok {
			item.SalesCount += p.SalesCount
			item.Revenue += p.SalesCount * p.SalePrice
		} else {
			levelAgg[p.LevelCode] = &vo.LevelSalesStatItem{
				LevelCode:  p.LevelCode,
				LevelName:  getLevelName(p.LevelCode),
				SalesCount: p.SalesCount,
				Revenue:    p.SalesCount * p.SalePrice,
			}
		}

		if item, ok := periodAgg[p.Period]; ok {
			item.SalesCount += p.SalesCount
			item.Revenue += p.SalesCount * p.SalePrice
		} else {
			periodAgg[p.Period] = &vo.PeriodSalesStatItem{
				Period:     p.Period,
				PeriodName: getPeriodName(p.Period),
				SalesCount: p.SalesCount,
				Revenue:    p.SalesCount * p.SalePrice,
			}
		}
	}

	for _, item := range levelAgg {
		stats.LevelStats = append(stats.LevelStats, *item)
	}
	for _, item := range periodAgg {
		stats.PeriodStats = append(stats.PeriodStats, *item)
	}

	totalIssued, err := s.couponRepo.CountIssued(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询优惠券发放数失败", err)
	}
	totalUsed, err := s.couponRepo.CountUsed(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询优惠券使用数失败", err)
	}
	stats.CouponStats.TotalIssued = totalIssued
	stats.CouponStats.TotalUsed = totalUsed
	if totalIssued > 0 {
		stats.CouponStats.UsageRate = totalUsed * 100 / totalIssued
	}

	return stats, nil
}

func (s *PackageService) toPackageDetailVO(p *model.SysPackage, benefitMap map[string]*model.SysMemberBenefit) vo.PackageDetailVO {
	vo := vo.PackageDetailVO{
		ID:            p.ID,
		Name:          p.Name,
		LevelCode:     p.LevelCode,
		LevelName:     getLevelName(p.LevelCode),
		Period:        p.Period,
		PeriodDays:    p.PeriodDays,
		OriginalPrice: p.OriginalPrice,
		SalePrice:     p.SalePrice,
		DailyPrice:    calcDailyPrice(p.SalePrice, p.PeriodDays),
		Description:   p.Description,
		Benefits:      make(map[string]int),
		SalesCount:    p.SalesCount,
	}

	benefit := benefitMap[p.LevelCode]
	if benefit != nil {
		vo.Benefits["monthlyDehazeQuota"] = benefit.MonthlyDehazeQuota
		vo.Benefits["monthlyEvaluateQuota"] = benefit.MonthlyEvaluateQuota
		vo.Benefits["historyRetention"] = benefit.HistoryRetention
		vo.Benefits["batchLimit"] = benefit.BatchLimit
		vo.Benefits["priority"] = int(benefit.Priority)
		vo.Benefits["advancedParams"] = int(benefit.AdvancedParams)
		vo.Benefits["hdExport"] = int(benefit.HdExport)
		vo.Benefits["reportExport"] = int(benefit.ReportExport)
		vo.Benefits["batchDownload"] = int(benefit.BatchDownload)
	}

	if p.BenefitOverrides != "" {
		var overrides map[string]int
		if err := json.Unmarshal([]byte(p.BenefitOverrides), &overrides); err == nil {
			for k, v := range overrides {
				vo.Benefits[k] = v
			}
		}
	}

	return vo
}

func getLevelName(levelCode string) string {
	if name, ok := levelNames[levelCode]; ok {
		return name
	}
	return levelCode
}

func getPeriodName(period string) string {
	if name, ok := periodNames[period]; ok {
		return name
	}
	return period
}

func calcDailyPrice(salePrice int64, periodDays int) int64 {
	if periodDays <= 0 {
		return 0
	}
	return salePrice / int64(periodDays)
}

func intPtr(v int) *int {
	return &v
}

func isCouponApplicable(c *model.SysCoupon, packageID int64) bool {
	if c.ApplicableScope == "" {
		return true
	}
	var scope []int64
	if err := json.Unmarshal([]byte(c.ApplicableScope), &scope); err != nil {
		return true
	}
	if len(scope) == 0 {
		return true
	}
	for _, id := range scope {
		if id == packageID {
			return true
		}
	}
	return false
}

func calcCouponAmount(c *model.SysCoupon, payableAmount int64) int64 {
	switch c.Type {
	case "full_reduction":
		if c.Threshold != nil && payableAmount >= *c.Threshold {
			return c.FaceValue
		}
		return 0
	case "discount":
		if c.FaceValue > 0 && c.FaceValue < 100 {
			return payableAmount * (100 - c.FaceValue) / 100
		}
		return 0
	case "no_threshold":
		return c.FaceValue
	case "trial":
		return payableAmount
	}
	return 0
}

var _ IPackageService = (*PackageService)(nil)

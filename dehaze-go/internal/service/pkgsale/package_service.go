package pkgsale

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	memberrepo "github.com/earthyzinc/dehaze-go/internal/repository/member"
	pkgsalerepo "github.com/earthyzinc/dehaze-go/internal/repository/pkgsale"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"gorm.io/gorm"
)

const (
	timeFormat         = "2006-01-02 15:04:05"
	packageDetailCacheTTL = 10 * time.Minute
)

var levelNames = map[string]string{
	"level_0": "普通用户",
	"level_1": "VIP1",
	"level_2": "VIP2",
	"level_3": "SVIP",
}

var periodNames = map[string]string{
	"monthly":   "月卡",
	"quarterly": "季卡",
	"yearly":    "年卡",
}

var validPeriods = map[string]bool{
	"monthly":   true,
	"quarterly": true,
	"yearly":    true,
}

type PackageService struct {
	db             *gorm.DB
	packageRepo    pkgsalerepo.IPackageRepository
	couponRepo     pkgsalerepo.ICouponRepository
	userCouponRepo pkgsalerepo.IUserCouponRepository
	benefitRepo    memberrepo.IMemberBenefitRepository
	cache          types.ICache
}

func NewPackageService(
	db *gorm.DB,
	packageRepo pkgsalerepo.IPackageRepository,
	couponRepo pkgsalerepo.ICouponRepository,
	userCouponRepo pkgsalerepo.IUserCouponRepository,
	benefitRepo memberrepo.IMemberBenefitRepository,
	cache types.ICache,
) *PackageService {
	return &PackageService{
		db:             db,
		packageRepo:    packageRepo,
		couponRepo:     couponRepo,
		userCouponRepo: userCouponRepo,
		benefitRepo:    benefitRepo,
		cache:          cache,
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
	cacheKey := fmt.Sprintf("package:detail:%d", id)
	if s.cache != nil {
		if cached, err := s.cache.Get(ctx, cacheKey); err == nil && cached != "" {
			var detail vo.PackageDetailVO
			if err := json.Unmarshal([]byte(cached), &detail); err == nil && detail.ID > 0 {
				return &detail, nil
			}
		}
	}

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

	detail := s.toPackageDetailVO(p, benefitMap)

	if s.cache != nil {
		if data, err := json.Marshal(detail); err == nil {
			_ = s.cache.Set(ctx, cacheKey, string(data), packageDetailCacheTTL)
		}
	}
	return &detail, nil
}

func (s *PackageService) invalidatePackageCache(ctx context.Context, id int64) {
	if s.cache == nil {
		return
	}
	_ = s.cache.Delete(ctx, fmt.Sprintf("package:detail:%d", id))
}

func (s *PackageService) CalculatePrice(ctx context.Context, userID, packageID int64, userCouponID *int64) (*vo.PriceResult, error) {
	p, err := s.packageRepo.FindByID(ctx, packageID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询套餐失败", err)
	}
	if p == nil {
		return nil, common.NewBizError(common.PACKAGE_NOT_FOUND, "套餐不存在")
	}

	salePrice := p.SalePrice
	discountAmount := s.calculatePromotionDiscount(ctx, p)
	couponAmount := int64(0)

	if userCouponID != nil && *userCouponID > 0 {
		uc, err := s.userCouponRepo.FindByUserIDAndStatusForUpdate(ctx, userID, *userCouponID)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询优惠券失败", err)
		}
		if uc == nil {
			return nil, common.NewBizError(common.COUPON_NOT_FOUND, "优惠券不存在")
		}
		if uc.Status != 1 && uc.Status != 4 {
			return nil, common.NewBizError(common.COUPON_ALREADY_USED, "优惠券已使用")
		}
		if uc.ExpireTime != nil && uc.ExpireTime.Before(time.Now()) {
			return nil, common.NewBizError(common.COUPON_EXPIRED, "优惠券已过期")
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

		couponBase := salePrice - discountAmount
		couponAmount = calcCouponAmount(c, couponBase)
		if couponAmount > couponBase {
			couponAmount = couponBase
		}
	}

	payable := salePrice - discountAmount - couponAmount
	if payable < 0 {
		payable = 0
	}

	return &vo.PriceResult{
		OriginalPrice:  p.OriginalPrice,
		DiscountAmount: discountAmount,
		CouponAmount:   couponAmount,
		PayableAmount:  payable,
	}, nil
}

func (s *PackageService) calculatePromotionDiscount(ctx context.Context, p *model.SysPackage) int64 {
	rows, err := s.packageRepo.FindActivePromotionsByPackageID(ctx, p.ID)
	if err != nil || len(rows) == 0 {
		return 0
	}
	now := time.Now()
	maxDiscount := int64(0)
	for _, pp := range rows {
		if pp.Status != 1 {
			continue
		}
		if now.Before(pp.StartTime) || now.After(pp.EndTime) {
			continue
		}
		var discount int64
		if pp.DiscountType == "percent" {
			discount = p.SalePrice * pp.DiscountValue / 100
		} else if pp.DiscountType == "fixed" {
			discount = pp.DiscountValue
		}
		if discount > maxDiscount {
			maxDiscount = discount
		}
	}
	return maxDiscount
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

	if p.BenefitOverrides.Valid {
		var overrides bo.BenefitOverrides
		if err := json.Unmarshal([]byte(p.BenefitOverrides.String), &overrides); err == nil {
			form.BenefitOverrides = &overrides
		}
	}

	return form, nil
}

func validatePackageForm(form *bo.PackageForm) error {
	if form.Name == "" {
		return common.NewBizError(common.PARAM_ERROR, "套餐名称不能为空")
	}
	if form.OriginalPrice < 1 {
		return common.NewBizError(common.PARAM_ERROR, "原价必须大于0")
	}
	if form.SalePrice < 1 {
		return common.NewBizError(common.PARAM_ERROR, "促销价必须大于0")
	}
	if form.SalePrice > form.OriginalPrice {
		return common.NewBizError(common.PARAM_ERROR, "促销价不能高于原价")
	}
	if !validPeriods[form.Period] {
		return common.NewBizError(common.PARAM_ERROR, "计费周期非法")
	}
	return nil
}

func (s *PackageService) Create(ctx context.Context, form *bo.PackageForm) error {
	if err := validatePackageForm(form); err != nil {
		return err
	}

	existing, err := s.packageRepo.FindByName(ctx, form.Name)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询套餐失败", err)
	}
	if existing != nil {
		return common.NewBizError(common.DATA_EXISTS, "套餐名称已存在")
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
		SalesCount:    0,
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
			p.BenefitOverrides = sql.NullString{String: string(data), Valid: true}
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
	if err := validatePackageForm(form); err != nil {
		return err
	}

	if p.Name != form.Name {
		dup, err := s.packageRepo.FindByName(ctx, form.Name)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询套餐失败", err)
		}
		if dup != nil && dup.ID != id {
			return common.NewBizError(common.DATA_EXISTS, "套餐名称已存在")
		}
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

	if form.BenefitOverrides != nil {
		data, err := json.Marshal(form.BenefitOverrides)
		if err == nil {
			updates["benefit_overrides"] = sql.NullString{String: string(data), Valid: true}
		}
	} else {
		updates["benefit_overrides"] = nil
	}

	if err := s.packageRepo.Update(ctx, id, updates); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新套餐失败", err)
	}
	s.invalidatePackageCache(ctx, id)
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

	if status == 0 {
		activePromos, err := s.packageRepo.FindActivePromotionsByPackageID(ctx, id)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询套餐促销活动失败", err)
		}
		if len(activePromos) > 0 {
			return common.NewBizError(common.PACKAGE_IN_PROMOTION, "套餐参与进行中促销活动，无法下架")
		}
	}

	if err := s.packageRepo.UpdateStatus(ctx, id, int8(status)); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新套餐状态失败", err)
	}
	s.invalidatePackageCache(ctx, id)
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
	for _, id := range ids {
		s.invalidatePackageCache(ctx, id)
	}
	return nil
}

func (s *PackageService) GetSalesStats(ctx context.Context) (*vo.SalesStatsVO, error) {
	stats := &vo.SalesStatsVO{
		PackageStats: make([]vo.PackageSalesStatItem, 0),
		LevelStats:   make([]vo.LevelSalesStatItem, 0),
		PeriodStats:  make([]vo.PeriodSalesStatItem, 0),
		CouponStats:  vo.CouponStatsVO{},
	}

	paidStatuses := []int8{2, 3}
	totalSales, err := s.packageRepo.CountOrdersByStatus(ctx, paidStatuses)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询已支付订单数失败", err)
	}
	stats.TotalSales = totalSales

	totalRevenue, err := s.packageRepo.SumPaidAmountByStatus(ctx, paidStatuses)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询订单收入失败", err)
	}
	stats.TotalRevenue = totalRevenue

	pkgRows, err := s.packageRepo.GetPackageOrderStats(ctx, paidStatuses)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询套餐销售统计失败", err)
	}
	for _, row := range pkgRows {
		stats.PackageStats = append(stats.PackageStats, vo.PackageSalesStatItem{
			PackageID:   row.PackageID,
			PackageName: row.PackageName,
			SalesCount:  row.Count,
			Revenue:     row.Revenue,
		})
	}

	levelRows, err := s.packageRepo.GetLevelOrderStats(ctx, paidStatuses)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询等级销售统计失败", err)
	}
	for _, row := range levelRows {
		stats.LevelStats = append(stats.LevelStats, vo.LevelSalesStatItem{
			LevelCode:  row.PackageLevel,
			LevelName:  getLevelName(row.PackageLevel),
			SalesCount: row.Count,
			Revenue:    row.Revenue,
		})
	}

	periodRows, err := s.packageRepo.GetPeriodOrderStats(ctx, paidStatuses)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询周期销售统计失败", err)
	}
	for _, row := range periodRows {
		stats.PeriodStats = append(stats.PeriodStats, vo.PeriodSalesStatItem{
			Period:     row.Period,
			PeriodName: getPeriodName(row.Period),
			SalesCount: row.Count,
			Revenue:    row.Revenue,
		})
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
		stats.CouponStats.UsageRate = float64(totalUsed) / float64(totalIssued)
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

	if p.BenefitOverrides.Valid {
		var overrides map[string]int
		if err := json.Unmarshal([]byte(p.BenefitOverrides.String), &overrides); err == nil {
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
	return (2*salePrice + int64(periodDays)) / (2 * int64(periodDays))
}

func intPtr(v int) *int {
	return &v
}

func isCouponApplicable(c *model.SysCoupon, packageID int64) bool {
	if !c.ApplicableScope.Valid || c.ApplicableScope.String == "" {
		return true
	}
	var scope []int64
	if err := json.Unmarshal([]byte(c.ApplicableScope.String), &scope); err != nil {
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

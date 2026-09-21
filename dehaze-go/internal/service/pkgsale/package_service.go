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
	timeFormat            = "2006-01-02 15:04:05"
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
	// 用户端详情仅提供在售套餐（T-PM-004）；后台编辑走 form 端点不受影响。
	// 缓存路径无此问题：上下架/修改/删除均会失效 package:detail 缓存。
	if p.Status != 1 {
		return nil, common.NewBizError(common.PACKAGE_OFF_SHELF, "套餐已下架")
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

	// 详情附进行中促销活动（python get_detail 同口径）；在售列表复用 VO 但不填充该字段
	detail.ActivePromotions = s.findActivePromotionVOs(ctx, p.ID)

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
	discountAmount, err := s.calculatePromotionDiscount(ctx, p, userID)
	if err != nil {
		return nil, err
	}
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
		if c == nil || c.Status != 1 {
			return nil, common.NewBizError(common.COUPON_NOT_FOUND, "优惠券不存在")
		}

		if !isCouponApplicable(c, packageID, p.PackageType) {
			return nil, common.NewBizError(common.COUPON_NOT_APPLICABLE, "优惠券不适用于该套餐")
		}

		// 体验券直接激活权益、不产生订单，不参与下单价格计算
		if c.Type == "trial" {
			return nil, common.NewBizError(common.BUSINESS_ERROR, "体验券不参与价格计算，请通过激活流程使用")
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

// calculatePromotionDiscount 汇总进行中促销的最大优惠（python calculate_price 同口径）。
// 新用户专享活动：当前用户已有已支付订单时整体拒绝下单。
// findActivePromotionVOs 查询套餐当前进行中的促销活动（状态启用且处于时间窗口内）
func (s *PackageService) findActivePromotionVOs(ctx context.Context, packageID int64) []vo.PromotionVO {
	rows, err := s.packageRepo.FindActivePromotionsByPackageID(ctx, packageID)
	if err != nil {
		return []vo.PromotionVO{}
	}
	now := time.Now()
	vos := make([]vo.PromotionVO, 0, len(rows))
	for _, pp := range rows {
		if pp.Status != 1 || now.Before(pp.StartTime) || now.After(pp.EndTime) {
			continue
		}
		vos = append(vos, vo.PromotionVO{
			ID:            pp.PromotionID,
			Name:          pp.Name,
			Type:          pp.Type,
			Description:   pp.Description,
			StartTime:     pp.StartTime.Format(timeFormat),
			EndTime:       pp.EndTime.Format(timeFormat),
			ActivityRules: unmarshalNullableRules(pp.ActivityRules),
			NewUserOnly:   int(pp.NewUserOnly),
			Status:        int(pp.Status),
		})
	}
	return vos
}

func (s *PackageService) calculatePromotionDiscount(ctx context.Context, p *model.SysPackage, userID int64) (int64, error) {
	rows, err := s.packageRepo.FindActivePromotionsByPackageID(ctx, p.ID)
	if err != nil || len(rows) == 0 {
		return 0, nil
	}
	now := time.Now()
	maxDiscount := int64(0)
	newUserOnly := false
	for _, pp := range rows {
		if pp.Status != 1 {
			continue
		}
		if now.Before(pp.StartTime) || now.After(pp.EndTime) {
			continue
		}
		if pp.NewUserOnly == 1 {
			newUserOnly = true
		}
		var discount int64
		switch pp.DiscountType {
		case "percent":
			discount = p.SalePrice * pp.DiscountValue / 100
		case "fixed":
			discount = pp.DiscountValue
		case "full_reduction":
			// 满减活动按规则 tiers 取满足门槛的最大面值
			rules := unmarshalNullableRules(pp.ActivityRules)
			if rules != nil {
				if tiers, ok := rules["tiers"].([]interface{}); ok {
					for _, t := range tiers {
						tier, ok := t.(map[string]interface{})
						if !ok {
							continue
						}
						threshold, _ := tier["threshold"].(float64)
						faceValue, _ := tier["faceValue"].(float64)
						if p.SalePrice >= int64(threshold) && int64(faceValue) > discount {
							discount = int64(faceValue)
						}
					}
				}
			}
		}
		if discount > maxDiscount {
			maxDiscount = discount
		}
	}

	if newUserOnly && userID > 0 {
		paidCount, err := s.packageRepo.CountPaidOrdersByUser(ctx, userID)
		if err != nil {
			return 0, common.WrapBizError(common.DATABASE_ERROR, "查询用户订单失败", err)
		}
		if paidCount > 0 {
			return 0, common.NewBizError(common.BUSINESS_ERROR, "该套餐仅限新用户购买")
		}
	}
	return maxDiscount, nil
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
			PackageType:   p.PackageType,
			LevelCode:     p.LevelCode.String,
			LevelName:     getLevelName(p.LevelCode.String),
			Period:        p.Period.String,
			PeriodDays:    int(p.PeriodDays.Int64),
			OriginalPrice: p.OriginalPrice,
			SalePrice:     p.SalePrice,
			DailyPrice:    calcDailyPrice(p.SalePrice, int(p.PeriodDays.Int64)),
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
		PackageType:   p.PackageType,
		LevelCode:     p.LevelCode.String,
		Period:        p.Period.String,
		OriginalPrice: p.OriginalPrice,
		SalePrice:     p.SalePrice,
		Description:   p.Description,
		Sort:          &p.Sort,
		Status:        intPtr(int(p.Status)),
	}
	if p.PeriodDays.Valid {
		days := int(p.PeriodDays.Int64)
		form.PeriodDays = &days
	}
	if p.CreditAmount.Valid {
		amount := p.CreditAmount.Int64
		form.CreditAmount = &amount
	}

	if p.BenefitOverrides.Valid {
		var overrides bo.BenefitOverrides
		if err := json.Unmarshal([]byte(p.BenefitOverrides.String), &overrides); err == nil {
			form.BenefitOverrides = &overrides
		}
	}

	return form, nil
}

// validatePackageForm 与 python _validate_package_form 对齐：vip/credit 差异字段条件必填
func validatePackageForm(form *bo.PackageForm, packageType string) error {
	if packageType != "vip" && packageType != "credit" {
		return common.NewBizError(common.PARAM_ERROR, "商品类型非法")
	}
	if form.SalePrice > form.OriginalPrice {
		return common.NewBizError(common.PARAM_ERROR, "促销价不能高于原价")
	}
	if packageType == "vip" {
		if form.LevelCode == "" || form.Period == "" || form.PeriodDays == nil {
			return common.NewBizError(common.PARAM_ERROR, "会员卡必须设置等级/周期/有效期")
		}
		if !validPeriods[form.Period] {
			return common.NewBizError(common.PARAM_ERROR, "计费周期非法")
		}
	} else {
		if form.CreditAmount == nil || *form.CreditAmount <= 0 {
			return common.NewBizError(common.PARAM_ERROR, "积分卡可得积分必须大于0")
		}
	}
	return nil
}

func (s *PackageService) Create(ctx context.Context, form *bo.PackageForm) error {
	packageType := form.PackageType
	if packageType == "" {
		packageType = "vip"
	}
	// 先做类型/参数校验，非法类型不应误报名称占用（文档 §5.1 顺序）
	if err := validatePackageForm(form, packageType); err != nil {
		return err
	}

	exists, err := s.packageRepo.ExistsByName(ctx, form.Name)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询套餐失败", err)
	}
	if exists {
		return common.NewBizError(common.DATA_EXISTS, "套餐名称已存在")
	}

	p := &model.SysPackage{
		Name:          form.Name,
		PackageType:   packageType,
		OriginalPrice: form.OriginalPrice,
		SalePrice:     form.SalePrice,
		Description:   form.Description,
		Sort:          0,
		SalesCount:    0,
		Status:        0,
	}
	if packageType == "credit" {
		p.CreditAmount = sql.NullInt64{Int64: *form.CreditAmount, Valid: true}
	} else {
		p.LevelCode = sql.NullString{String: form.LevelCode, Valid: true}
		p.Period = sql.NullString{String: form.Period, Valid: true}
		p.PeriodDays = sql.NullInt64{Int64: int64(*form.PeriodDays), Valid: true}
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
	// 商品类型创建后锁定，以库中记录为准，请求携带的 packageType 被忽略
	if err := validatePackageForm(form, p.PackageType); err != nil {
		return err
	}

	if p.Name != form.Name {
		exists, err := s.packageRepo.ExistsByName(ctx, form.Name, id)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "检查套餐名称是否存在失败", err)
		}
		if exists {
			return common.NewBizError(common.DATA_EXISTS, "套餐名称已存在")
		}
	}

	updates := map[string]interface{}{
		"name":           form.Name,
		"original_price": form.OriginalPrice,
		"sale_price":     form.SalePrice,
		"description":    form.Description,
	}
	if p.PackageType == "credit" {
		updates["credit_amount"] = sql.NullInt64{Int64: *form.CreditAmount, Valid: true}
		updates["level_code"] = sql.NullString{}
		updates["period"] = sql.NullString{}
		updates["period_days"] = sql.NullInt64{}
	} else {
		updates["level_code"] = sql.NullString{String: form.LevelCode, Valid: true}
		updates["period"] = sql.NullString{String: form.Period, Valid: true}
		updates["period_days"] = sql.NullInt64{Int64: int64(*form.PeriodDays), Valid: true}
		updates["credit_amount"] = sql.NullInt64{}
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
		// 促销进行中（启用且处于时间窗口内）不允许下架（T-PM-039）
		now := time.Now()
		rows, err := s.packageRepo.FindActivePromotionsByPackageID(ctx, id)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询套餐促销活动失败", err)
		}
		for _, pp := range rows {
			if pp.Status == 1 && !now.Before(pp.StartTime) && !now.After(pp.EndTime) {
				return common.NewBizError(common.PACKAGE_IN_PROMOTION, "套餐参与进行中促销活动，无法下架")
			}
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
		PackageType:   p.PackageType,
		LevelCode:     p.LevelCode.String,
		LevelName:     getLevelName(p.LevelCode.String),
		Period:        p.Period.String,
		PeriodDays:    int(p.PeriodDays.Int64),
		OriginalPrice: p.OriginalPrice,
		SalePrice:     p.SalePrice,
		DailyPrice:    calcDailyPrice(p.SalePrice, int(p.PeriodDays.Int64)),
		Description:   p.Description,
		Benefits:      make(map[string]int),
		SalesCount:    p.SalesCount,
	}
	if p.CreditAmount.Valid {
		amount := p.CreditAmount.Int64
		vo.CreditAmount = &amount
		vo.CreditUnitPrice = calcCreditUnitPrice(p.SalePrice, amount)
	}

	benefit := benefitMap[p.LevelCode.String]
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

// isCouponApplicable 适用范围支持商品 ID 与商品类型 vip/credit（python calculate_price 同口径）
func isCouponApplicable(c *model.SysCoupon, packageID int64, packageType string) bool {
	if !c.ApplicableScope.Valid || c.ApplicableScope.String == "" {
		return true
	}
	var scopes []interface{}
	if err := json.Unmarshal([]byte(c.ApplicableScope.String), &scopes); err != nil {
		return true
	}
	if len(scopes) == 0 {
		return true
	}
	for _, s := range scopes {
		switch v := s.(type) {
		case float64:
			if int64(v) == packageID {
				return true
			}
		case string:
			if v == packageType {
				return true
			}
		}
	}
	return false
}

func calcCreditUnitPrice(salePrice, creditAmount int64) int64 {
	if creditAmount <= 0 {
		return 0
	}
	return salePrice / creditAmount
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
	}
	return 0
}

var _ IPackageService = (*PackageService)(nil)

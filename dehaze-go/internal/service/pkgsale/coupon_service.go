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
	pkgsalerepo "github.com/earthyzinc/dehaze-go/internal/repository/pkgsale"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"gorm.io/gorm"
)

type CouponService struct {
	db             *gorm.DB
	couponRepo     pkgsalerepo.ICouponRepository
	userCouponRepo pkgsalerepo.IUserCouponRepository
	cache          types.ICache
}

func NewCouponService(
	db *gorm.DB,
	couponRepo pkgsalerepo.ICouponRepository,
	userCouponRepo pkgsalerepo.IUserCouponRepository,
	cache types.ICache,
) *CouponService {
	return &CouponService{
		db:             db,
		couponRepo:     couponRepo,
		userCouponRepo: userCouponRepo,
		cache:          cache,
	}
}

func (s *CouponService) ListMy(ctx context.Context, userID int64, status *int) ([]vo.UserCouponVO, error) {
	list, err := s.userCouponRepo.FindByUserID(ctx, userID, status)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询我的优惠券失败", err)
	}

	couponIDs := make([]int64, 0, len(list))
	for _, uc := range list {
		couponIDs = append(couponIDs, uc.CouponID)
	}
	couponMap := make(map[int64]*model.SysCoupon)
	if len(couponIDs) > 0 {
		coupons, err := s.couponRepo.FindByIDsIncludeDeleted(ctx, couponIDs)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询优惠券模板失败", err)
		}
		for i := range coupons {
			couponMap[coupons[i].ID] = &coupons[i]
		}
	}

	vos := make([]vo.UserCouponVO, 0, len(list))
	for _, uc := range list {
		v := vo.UserCouponVO{
			ID:          uc.ID,
			CouponID:    uc.CouponID,
			Status:      int(uc.Status),
			ReceiveTime: uc.ReceiveTime.Format(timeFormat),
			UsedOrderID: uc.UsedOrderID,
		}
		if uc.ExpireTime != nil {
			t := uc.ExpireTime.Format(timeFormat)
			v.ExpireTime = &t
		}
		if uc.UsedTime != nil {
			t := uc.UsedTime.Format(timeFormat)
			v.UsedTime = &t
		}

		if c, ok := couponMap[uc.CouponID]; ok {
			v.CouponName = c.Name
			v.Type = c.Type
			v.FaceValue = c.FaceValue
			v.Threshold = c.Threshold
			v.ApplicableScope = parseScope(c.ApplicableScope.String)
		}

		vos = append(vos, v)
	}
	return vos, nil
}

func (s *CouponService) Receive(ctx context.Context, userID, couponID int64) (*vo.CouponReceiveResult, error) {
	c, err := s.couponRepo.FindByID(ctx, couponID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询优惠券失败", err)
	}
	if c == nil {
		return nil, common.NewBizError(common.COUPON_NOT_FOUND, "优惠券不存在")
	}

	if c.Status != 1 {
		return nil, common.NewBizError(common.COUPON_STOCK_EMPTY, "优惠券已停用")
	}

	if c.TotalQty > 0 && c.IssuedQty >= c.TotalQty {
		return nil, common.NewBizError(common.COUPON_STOCK_EMPTY, "优惠券已领完")
	}

	count, err := s.userCouponRepo.CountByUserIDAndCouponID(ctx, userID, couponID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询领取记录失败", err)
	}
	if count >= int64(c.PerUserLimit) {
		return nil, common.NewBizError(common.BUSINESS_ERROR, "已超过每人限领数量")
	}

	if s.cache != nil {
		rateKey := fmt.Sprintf("coupon:receive:rate:%d", userID)
		rateCount, err := s.cache.Incr(ctx, rateKey)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "频率限制检查失败", err)
		}
		if rateCount == 1 {
			_, _ = s.cache.Expire(ctx, rateKey, 60*time.Second)
		}
		if rateCount > 5 {
			return nil, common.NewBizError(common.RATE_LIMIT, "操作过于频繁，请稍后再试")
		}
	}

	uc := &model.SysUserCoupon{
		UserID:      userID,
		CouponID:    couponID,
		Status:      1,
		ReceiveTime: time.Now(),
	}

	if c.ValidType == "relative" && c.ValidDays != nil && *c.ValidDays > 0 {
		expire := time.Now().AddDate(0, 0, *c.ValidDays)
		uc.ExpireTime = &expire
	} else if c.ValidType == "fixed" && c.ValidEnd != nil {
		uc.ExpireTime = c.ValidEnd
	}

	err = s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txUserCouponRepo := pkgsalerepo.NewUserCouponRepository(tx)
		txCouponRepo := pkgsalerepo.NewCouponRepository(tx)

		if err := txUserCouponRepo.Create(ctx, uc); err != nil {
			return err
		}
		if err := txCouponRepo.IncrementIssuedQty(ctx, couponID); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "领取优惠券失败", err)
	}

	return &vo.CouponReceiveResult{UserCouponID: uc.ID}, nil
}

func (s *CouponService) GetPage(ctx context.Context, q *query.CouponPageQuery) (*vo.PageResult[vo.CouponVO], error) {
	list, total, err := s.couponRepo.FindPage(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询优惠券列表失败", err)
	}
	vos := make([]vo.CouponVO, 0, len(list))
	for _, c := range list {
		v := vo.CouponVO{
			ID:           c.ID,
			Name:         c.Name,
			Type:         c.Type,
			FaceValue:    c.FaceValue,
			Threshold:    c.Threshold,
			ValidType:    c.ValidType,
			ValidDays:    c.ValidDays,
			TotalQty:     c.TotalQty,
			IssuedQty:    c.IssuedQty,
			UsedQty:      c.UsedQty,
			PerUserLimit: c.PerUserLimit,
			ApplicableScope: parseScope(c.ApplicableScope.String),
			Status:       int(c.Status),
			CreateTime:   c.CreatedAt.Format(timeFormat),
		}
		if c.ValidStart != nil {
			t := c.ValidStart.Format(timeFormat)
			v.ValidStart = &t
		}
		if c.ValidEnd != nil {
			t := c.ValidEnd.Format(timeFormat)
			v.ValidEnd = &t
		}
		vos = append(vos, v)
	}
	return &vo.PageResult[vo.CouponVO]{List: vos, Total: total}, nil
}

func (s *CouponService) Create(ctx context.Context, form *bo.CouponForm) (*vo.CouponCreateResult, error) {
	if form.Name == "" {
		return nil, common.NewBizError(common.PARAM_ERROR, "优惠券名称不能为空")
	}
	if form.TotalQty < -1 {
		return nil, common.NewBizError(common.PARAM_ERROR, "库存不能为负数")
	}

	c := &model.SysCoupon{
		Name:         form.Name,
		Type:         form.Type,
		FaceValue:    form.FaceValue,
		Threshold:    form.Threshold,
		ValidType:    form.ValidType,
		TotalQty:     form.TotalQty,
		PerUserLimit: form.PerUserLimit,
		Status:       1,
	}

	if form.ValidStart != nil && *form.ValidStart != "" {
		t, err := time.ParseInLocation(timeFormat, *form.ValidStart, time.Local)
		if err == nil {
			c.ValidStart = &t
		}
	}
	if form.ValidEnd != nil && *form.ValidEnd != "" {
		t, err := time.ParseInLocation(timeFormat, *form.ValidEnd, time.Local)
		if err == nil {
			c.ValidEnd = &t
		}
	}
	if form.ValidDays != nil {
		c.ValidDays = form.ValidDays
	}
	if form.Status != nil {
		c.Status = int8(*form.Status)
	}
	if form.ApplicableScope != nil && len(form.ApplicableScope) > 0 {
		data, err := json.Marshal(form.ApplicableScope)
		if err == nil {
			c.ApplicableScope = sql.NullString{String: string(data), Valid: true}
		}
	}

	if err := s.couponRepo.Create(ctx, c); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建优惠券失败", err)
	}
	return &vo.CouponCreateResult{ID: c.ID}, nil
}

func (s *CouponService) Update(ctx context.Context, id int64, form *bo.CouponForm) error {
	c, err := s.couponRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询优惠券失败", err)
	}
	if c == nil {
		return common.NewBizError(common.COUPON_NOT_FOUND, "优惠券不存在")
	}

	updates := map[string]interface{}{
		"name":           form.Name,
		"type":           form.Type,
		"face_value":     form.FaceValue,
		"valid_type":     form.ValidType,
		"total_qty":      form.TotalQty,
		"per_user_limit": form.PerUserLimit,
	}

	if form.Threshold != nil {
		updates["threshold"] = *form.Threshold
	} else {
		updates["threshold"] = nil
	}
	if form.ValidStart != nil && *form.ValidStart != "" {
		t, err := time.ParseInLocation(timeFormat, *form.ValidStart, time.Local)
		if err == nil {
			updates["valid_start"] = t
		}
	}
	if form.ValidEnd != nil && *form.ValidEnd != "" {
		t, err := time.ParseInLocation(timeFormat, *form.ValidEnd, time.Local)
		if err == nil {
			updates["valid_end"] = t
		}
	}
	if form.ValidDays != nil {
		updates["valid_days"] = *form.ValidDays
	}
	if form.Status != nil {
		updates["status"] = int8(*form.Status)
	}
	if form.ApplicableScope != nil {
		if len(form.ApplicableScope) > 0 {
			data, err := json.Marshal(form.ApplicableScope)
			if err == nil {
				updates["applicable_scope"] = sql.NullString{String: string(data), Valid: true}
			}
		} else {
			updates["applicable_scope"] = nil
		}
	}

	if err := s.couponRepo.Update(ctx, id, updates); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新优惠券失败", err)
	}
	return nil
}

func (s *CouponService) DeleteByIDs(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return common.NewBizError(common.PARAM_ERROR, "请选择要删除的优惠券")
	}

	for _, id := range ids {
		c, err := s.couponRepo.FindByID(ctx, id)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询优惠券失败", err)
		}
		if c == nil {
			return common.NewBizError(common.COUPON_NOT_FOUND, fmt.Sprintf("优惠券(id=%d)不存在", id))
		}
	}

	usedCount, err := s.userCouponRepo.CountUsedByCouponIDs(ctx, ids)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询优惠券使用记录失败", err)
	}
	if usedCount > 0 {
		return common.NewBizError(common.DATA_BIND_EXISTS, "优惠券已发放使用，无法删除")
	}

	err = s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txCouponRepo := pkgsalerepo.NewCouponRepository(tx)
		txUserCouponRepo := pkgsalerepo.NewUserCouponRepository(tx)

		if err := txUserCouponRepo.DeleteByCouponIDs(ctx, ids); err != nil {
			return err
		}
		if err := txCouponRepo.DeleteByIDs(ctx, ids); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除优惠券失败", err)
	}
	return nil
}

func (s *CouponService) BatchDistribute(ctx context.Context, form *bo.CouponBatchDistributeForm) (*vo.CouponBatchDistributeResult, error) {
	c, err := s.couponRepo.FindByID(ctx, form.CouponID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询优惠券失败", err)
	}
	if c == nil {
		return nil, common.NewBizError(common.COUPON_NOT_FOUND, "优惠券不存在")
	}

	var userIDs []int64
	switch form.TargetScope {
	case "users":
		userIDs = form.UserIDs
	case "level":
		userIDs, err = s.findUserIDsByLevelCodes(ctx, form.LevelCodes)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询用户失败", err)
		}
	case "all":
		userIDs, err = s.findAllUserIDs(ctx)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询用户失败", err)
		}
	default:
		return nil, common.NewBizError(common.PARAM_ERROR, "无效的发放范围")
	}

	result := &vo.CouponBatchDistributeResult{}
	for _, uid := range userIDs {
		uc := &model.SysUserCoupon{
			UserID:      uid,
			CouponID:    form.CouponID,
			Status:      1,
			ReceiveTime: time.Now(),
		}
		if c.ValidType == "relative" && c.ValidDays != nil && *c.ValidDays > 0 {
			expire := time.Now().AddDate(0, 0, *c.ValidDays)
			uc.ExpireTime = &expire
		} else if c.ValidType == "fixed" && c.ValidEnd != nil {
			uc.ExpireTime = c.ValidEnd
		}

		if err := s.userCouponRepo.Create(ctx, uc); err != nil {
			result.FailCount++
			continue
		}
		if err := s.couponRepo.IncrementIssuedQty(ctx, form.CouponID); err != nil {
			result.FailCount++
			continue
		}
		result.SuccessCount++
	}

	return result, nil
}

func (s *CouponService) findUserIDsByLevelCodes(ctx context.Context, levelCodes []string) ([]int64, error) {
	if len(levelCodes) == 0 {
		return nil, nil
	}
	var ids []int64
	err := s.db.WithContext(ctx).
		Table("sys_member").
		Where("level_code IN ? AND deleted = 0 AND status = 1", levelCodes).
		Pluck("user_id", &ids).Error
	return ids, err
}

func (s *CouponService) findAllUserIDs(ctx context.Context) ([]int64, error) {
	var ids []int64
	err := s.db.WithContext(ctx).
		Table("sys_user").
		Where("deleted = 0 AND status = 1").
		Pluck("id", &ids).Error
	return ids, err
}

func parseScope(scope string) []int64 {
	if scope == "" {
		return nil
	}
	var ids []int64
	if err := json.Unmarshal([]byte(scope), &ids); err != nil {
		return nil
	}
	return ids
}

var _ ICouponService = (*CouponService)(nil)

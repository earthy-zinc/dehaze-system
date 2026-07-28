package order

import (
	"context"
	"fmt"
	"math/rand"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	memberrepo "github.com/earthyzinc/dehaze-go/internal/repository/member"
	orderrepo "github.com/earthyzinc/dehaze-go/internal/repository/order"
	pkgsalerepo "github.com/earthyzinc/dehaze-go/internal/repository/pkgsale"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"gorm.io/gorm"
)

const (
	orderExpireMinutes = 30
	timeFormat         = "2006-01-02 15:04:05"
)

type OrderService struct {
	db               *gorm.DB
	orderRepo        orderrepo.IOrderRepository
	paymentRepo      orderrepo.IPaymentRecordRepository
	refundRepo       orderrepo.IRefundRecordRepository
	autoRenewRepo    orderrepo.IAutoRenewRepository
	packageRepo      pkgsalerepo.IPackageRepository
	couponRepo       pkgsalerepo.ICouponRepository
	userCouponRepo   pkgsalerepo.IUserCouponRepository
	memberRepo       memberrepo.IMemberRepository
}

func NewOrderService(
	db *gorm.DB,
	orderRepo orderrepo.IOrderRepository,
	paymentRepo orderrepo.IPaymentRecordRepository,
	refundRepo orderrepo.IRefundRecordRepository,
	autoRenewRepo orderrepo.IAutoRenewRepository,
	packageRepo pkgsalerepo.IPackageRepository,
	couponRepo pkgsalerepo.ICouponRepository,
	userCouponRepo pkgsalerepo.IUserCouponRepository,
	memberRepo memberrepo.IMemberRepository,
) *OrderService {
	return &OrderService{
		db:            db,
		orderRepo:     orderRepo,
		paymentRepo:   paymentRepo,
		refundRepo:    refundRepo,
		autoRenewRepo: autoRenewRepo,
		packageRepo:   packageRepo,
		couponRepo:    couponRepo,
		userCouponRepo: userCouponRepo,
		memberRepo:    memberRepo,
	}
}

func (s *OrderService) Create(ctx context.Context, userID int64, form *bo.OrderCreateForm) (*vo.PayResult, error) {
	p, err := s.packageRepo.FindByID(ctx, form.PackageID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询套餐失败", err)
	}
	if p == nil {
		return nil, common.NewBizError(common.PACKAGE_NOT_FOUND, "套餐不存在")
	}
	if p.Status != 1 {
		return nil, common.NewBizError(common.PACKAGE_OFF_SHELF, "套餐已下架")
	}

	orderNo := generateOrderNo()
	now := time.Now()
	expireTime := now.Add(orderExpireMinutes * time.Minute)

	discountAmount := p.OriginalPrice - p.SalePrice
	couponAmount := int64(0)
	var userCouponID *int64

	if form.CouponID != nil && *form.CouponID > 0 {
		uc, err := s.userCouponRepo.FindByUserIDAndStatusForUpdate(ctx, userID, *form.CouponID)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询用户优惠券失败", err)
		}
		if uc == nil || uc.Status != 1 {
			return nil, common.NewBizError(common.COUPON_NOT_FOUND, "优惠券不存在或不可用")
		}

		c, err := s.couponRepo.FindByID(ctx, uc.CouponID)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询优惠券模板失败", err)
		}
		if c == nil {
			return nil, common.NewBizError(common.COUPON_NOT_FOUND, "优惠券不存在")
		}

		couponAmount = calcCouponAmount(c, p.SalePrice)
		if couponAmount > p.SalePrice {
			couponAmount = p.SalePrice
		}
		userCouponID = &uc.ID
	}

	payableAmount := p.SalePrice - couponAmount

	order := &model.SysOrder{
		OrderNo:        orderNo,
		UserID:         userID,
		PackageID:      p.ID,
		PackageName:    p.Name,
		PackageLevel:   p.LevelCode,
		PeriodDays:     p.PeriodDays,
		OriginalPrice:  p.OriginalPrice,
		DiscountAmount: discountAmount,
		CouponID:       userCouponID,
		CouponAmount:   couponAmount,
		PayableAmount:  payableAmount,
		PaidAmount:     0,
		Status:         1,
		ExpireTime:     expireTime,
		IsAutoRenew:    0,
	}

	paid := false
	payMethod := form.PayMethod

	err = s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txOrderRepo := orderrepo.NewOrderRepository(tx)
		txUserCouponRepo := pkgsalerepo.NewUserCouponRepository(tx)

		if err := txOrderRepo.Create(ctx, order); err != nil {
			return err
		}

		if userCouponID != nil {
			if err := txUserCouponRepo.Update(ctx, *userCouponID, map[string]interface{}{
				"status":        4,
				"used_order_id": order.ID,
			}); err != nil {
				return err
			}
		}

		if payMethod == "balance" {
			return s.completePaymentInTx(ctx, tx, order)
		}
		return nil
	})
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建订单失败", err)
	}

	if payMethod == "balance" {
		paid = true
		_ = s.packageRepo.IncrementSalesCount(ctx, p.ID, 1)
	}

	result := &vo.PayResult{
		OrderNo:   orderNo,
		PayMethod: payMethod,
		Paid:      paid,
	}
	if !paid && (payMethod == "wechat" || payMethod == "alipay") {
		result.PayURL = ""
		result.QRCode = ""
	}
	return result, nil
}

func (s *OrderService) completePaymentInTx(ctx context.Context, tx *gorm.DB, order *model.SysOrder) error {
	txOrderRepo := orderrepo.NewOrderRepository(tx)
	txPaymentRepo := orderrepo.NewPaymentRecordRepository(tx)
	txUserCouponRepo := pkgsalerepo.NewUserCouponRepository(tx)
	txCouponRepo := pkgsalerepo.NewCouponRepository(tx)
	txMemberRepo := memberrepo.NewMemberRepository(tx)

	now := time.Now()
	paymentNo := fmt.Sprintf("PAY%s%06d", now.Format("20060102150405"), rand.Intn(1000000))

	payment := &model.SysPaymentRecord{
		OrderID:      order.ID,
		UserID:       order.UserID,
		PaymentNo:    paymentNo,
		Channel:      "balance",
		Amount:       order.PayableAmount,
		Status:       2,
		CallbackTime: &now,
	}
	if err := txPaymentRepo.Create(ctx, payment); err != nil {
		return err
	}

	effectiveTime := now
	packageExpireTime := now.AddDate(0, 0, order.PeriodDays)

	member, _ := txMemberRepo.FindByUserID(ctx, order.UserID)
	if member != nil && member.ExpireTime != nil && member.ExpireTime.After(now) {
		effectiveTime = *member.ExpireTime
		packageExpireTime = effectiveTime.AddDate(0, 0, order.PeriodDays)
	}

	if err := txOrderRepo.Update(ctx, order.ID, map[string]interface{}{
		"status":              2,
		"paid_amount":         order.PayableAmount,
		"paid_time":           now,
		"pay_method":          "balance",
		"effective_time":      effectiveTime,
		"package_expire_time": packageExpireTime,
	}); err != nil {
		return err
	}

	if order.CouponID != nil && *order.CouponID > 0 {
		uc, _ := txUserCouponRepo.FindByID(ctx, *order.CouponID)
		if uc != nil {
			_ = txUserCouponRepo.Update(ctx, uc.ID, map[string]interface{}{
				"status":    2,
				"used_time": now,
			})
			_ = txCouponRepo.IncrementUsedQty(ctx, uc.CouponID)
		}
	}

	return s.updateMemberAfterPaymentInTx(ctx, txMemberRepo, order.UserID, order.PackageLevel, order.PaidAmount, &packageExpireTime)
}

func (s *OrderService) updateMemberAfterPaymentInTx(ctx context.Context, txMemberRepo memberrepo.IMemberRepository, userID int64, levelCode string, amount int64, expireTime *time.Time) error {
	member, _ := txMemberRepo.FindByUserID(ctx, userID)
	if member == nil {
		now := time.Now()
		newMember := &model.SysMember{
			UserID:           userID,
			LevelCode:        levelCode,
			LevelSource:      "package",
			TotalConsumption: amount,
			ExpireTime:       expireTime,
			BecomeMemberTime: &now,
			Status:           1,
		}
		return txMemberRepo.Update(ctx, userID, map[string]interface{}{
			"user_id":             newMember.UserID,
			"level_code":          newMember.LevelCode,
			"level_source":        newMember.LevelSource,
			"total_consumption":   newMember.TotalConsumption,
			"expire_time":         *expireTime,
			"become_member_time":  now,
			"status":              1,
		})
	}

	updates := map[string]interface{}{
		"level_code":        levelCode,
		"level_source":      "package",
		"total_consumption": member.TotalConsumption + amount,
		"expire_time":       *expireTime,
		"status":            1,
	}
	return txMemberRepo.Update(ctx, userID, updates)
}

func (s *OrderService) ListMy(ctx context.Context, userID int64, q *query.MyOrderQuery) (*vo.PageResult[vo.MyOrderVO], error) {
	list, total, err := s.orderRepo.FindPageMyOrders(ctx, userID, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询我的订单失败", err)
	}
	vos := make([]vo.MyOrderVO, 0, len(list))
	for _, o := range list {
		vos = append(vos, toMyOrderVO(&o.SysOrder))
	}
	return &vo.PageResult[vo.MyOrderVO]{List: vos, Total: total}, nil
}

func (s *OrderService) GetDetail(ctx context.Context, orderNo string) (*vo.OrderDetailVO, error) {
	o, err := s.orderRepo.FindByOrderNo(ctx, orderNo)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询订单失败", err)
	}
	if o == nil {
		return nil, common.NewBizError(common.ORDER_NOT_FOUND, "订单不存在")
	}

	detail := &vo.OrderDetailVO{
		OrderPageVO: vo.OrderPageVO{
			MyOrderVO:      toMyOrderVO(o),
			UserID:         o.UserID,
			OriginalPrice:  o.OriginalPrice,
			DiscountAmount: o.DiscountAmount,
			CouponAmount:   o.CouponAmount,
		},
		ExpireTime:   o.ExpireTime.Format(timeFormat),
		IsAutoRenew:  int(o.IsAutoRenew),
		CancelReason: o.CancelReason,
	}
	if o.EffectiveTime != nil {
		t := o.EffectiveTime.Format(timeFormat)
		detail.EffectiveTime = &t
	}

	payments, err := s.paymentRepo.FindByOrderID(ctx, o.ID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询支付记录失败", err)
	}
	paymentVOs := make([]vo.PaymentRecordVO, 0, len(payments))
	for _, p := range payments {
		pv := vo.PaymentRecordVO{
			ID:        p.ID,
			PaymentNo: p.PaymentNo,
			Channel:   p.Channel,
			Amount:    p.Amount,
			Status:    int(p.Status),
			CreateTime: p.CreateTime.Format(timeFormat),
		}
		if p.CallbackTime != nil {
			t := p.CallbackTime.Format(timeFormat)
			pv.CallbackTime = &t
		}
		paymentVOs = append(paymentVOs, pv)
	}
	detail.PaymentRecords = paymentVOs

	refund, err := s.refundRepo.FindByOrderID(ctx, o.ID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询退款记录失败", err)
	}
	if refund != nil {
		detail.RefundRecord = s.toRefundRecordVO(ctx, refund, "", "")
	}

	user, _ := s.findUsernameByUserID(ctx, o.UserID)
	detail.Username = user

	return detail, nil
}

func (s *OrderService) Cancel(ctx context.Context, orderNo string, reason string) error {
	o, err := s.orderRepo.FindByOrderNo(ctx, orderNo)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询订单失败", err)
	}
	if o == nil {
		return common.NewBizError(common.ORDER_NOT_FOUND, "订单不存在")
	}
	if o.Status != 1 {
		return common.NewBizError(common.ORDER_STATUS_INVALID, "订单状态不允许此操作")
	}

	return s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txOrderRepo := orderrepo.NewOrderRepository(tx)
		txUserCouponRepo := pkgsalerepo.NewUserCouponRepository(tx)

		if err := txOrderRepo.Update(ctx, o.ID, map[string]interface{}{
			"status":        4,
			"cancel_reason": reason,
		}); err != nil {
			return err
		}

		if o.CouponID != nil && *o.CouponID > 0 {
			_ = txUserCouponRepo.Update(ctx, *o.CouponID, map[string]interface{}{
				"status":        1,
				"used_order_id": nil,
			})
		}
		return nil
	})
}

func (s *OrderService) Pay(ctx context.Context, orderNo string, req *bo.PayRequest) (*vo.PayResult, error) {
	o, err := s.orderRepo.FindByOrderNo(ctx, orderNo)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询订单失败", err)
	}
	if o == nil {
		return nil, common.NewBizError(common.ORDER_NOT_FOUND, "订单不存在")
	}
	if o.Status != 1 {
		return nil, common.NewBizError(common.ORDER_STATUS_INVALID, "订单状态不允许此操作")
	}
	if time.Now().After(o.ExpireTime) {
		_ = s.orderRepo.Update(ctx, o.ID, map[string]interface{}{
			"status":        4,
			"cancel_reason": "支付超时自动取消",
		})
		return nil, common.NewBizError(common.ORDER_STATUS_INVALID, "订单已超时")
	}

	result := &vo.PayResult{
		OrderNo:   orderNo,
		PayMethod: req.PayMethod,
	}

	if req.PayMethod == "balance" {
		err = s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
			return s.completePaymentInTx(ctx, tx, o)
		})
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "支付失败", err)
		}
		result.Paid = true
		_ = s.packageRepo.IncrementSalesCount(ctx, o.PackageID, 1)
	} else {
		result.Paid = false
	}

	return result, nil
}

func (s *OrderService) ApplyRefund(ctx context.Context, userID int64, orderNo string, form *bo.RefundApplyForm) error {
	o, err := s.orderRepo.FindByOrderNo(ctx, orderNo)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询订单失败", err)
	}
	if o == nil {
		return common.NewBizError(common.ORDER_NOT_FOUND, "订单不存在")
	}
	if o.UserID != userID {
		return common.NewBizError(common.ORDER_NOT_FOUND, "订单不存在")
	}
	if o.Status != 2 && o.Status != 3 {
		return common.NewBizError(common.ORDER_STATUS_INVALID, "订单状态不允许此操作")
	}

	existing, _ := s.refundRepo.FindByOrderID(ctx, o.ID)
	if existing != nil {
		return common.NewBizError(common.ORDER_STATUS_INVALID, "订单已申请退款")
	}

	reason := form.Reason
	if form.CustomReason != "" {
		reason = form.CustomReason
	}

	refundNo := fmt.Sprintf("RF%s%06d", time.Now().Format("20060102150405"), rand.Intn(1000000))
	refund := &model.SysRefundRecord{
		RefundNo:     refundNo,
		OrderID:      o.ID,
		UserID:       userID,
		RefundAmount: o.PaidAmount,
		Reason:       reason,
		Status:       1,
	}

	return s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txRefundRepo := orderrepo.NewRefundRecordRepository(tx)
		txOrderRepo := orderrepo.NewOrderRepository(tx)

		if err := txRefundRepo.Create(ctx, refund); err != nil {
			return err
		}
		return txOrderRepo.Update(ctx, o.ID, map[string]interface{}{
			"status": 5,
		})
	})
}

func (s *OrderService) UpdateAutoRenewConfig(ctx context.Context, userID int64, form *bo.AutoRenewConfigForm) error {
	p, err := s.packageRepo.FindByID(ctx, form.PackageID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询套餐失败", err)
	}
	if p == nil {
		return common.NewBizError(common.PACKAGE_NOT_FOUND, "套餐不存在")
	}

	var status int8 = 0
	if form.Enabled {
		status = 1
	}

	var nextRenewTime *time.Time
	if form.Enabled {
		member, _ := s.memberRepo.FindByUserID(ctx, userID)
		base := time.Now()
		if member != nil && member.ExpireTime != nil && member.ExpireTime.After(base) {
			base = *member.ExpireTime
		}
		nextRenew := base
		nextRenewTime = &nextRenew
	}

	closeReason := ""
	if !form.Enabled {
		closeReason = "用户关闭自动续费"
	}

	ar := &model.SysAutoRenew{
		UserID:        userID,
		PackageID:     form.PackageID,
		PayMethod:     form.PayMethod,
		Status:        status,
		NextRenewTime: nextRenewTime,
		CloseReason:   closeReason,
	}

	if err := s.autoRenewRepo.Upsert(ctx, ar); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新自动续费配置失败", err)
	}
	return nil
}

func (s *OrderService) GetAutoRenewConfig(ctx context.Context, userID, packageID int64) (*vo.AutoRenewConfigVO, error) {
	p, err := s.packageRepo.FindByID(ctx, packageID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询套餐失败", err)
	}
	if p == nil {
		return nil, common.NewBizError(common.PACKAGE_NOT_FOUND, "套餐不存在")
	}

	ar, _ := s.autoRenewRepo.FindByUserIDAndPackageID(ctx, userID, packageID)

	vo := &vo.AutoRenewConfigVO{
		UserID:      userID,
		PackageID:   packageID,
		PackageName: p.Name,
	}
	if ar != nil {
		vo.PayMethod = ar.PayMethod
		vo.Enabled = ar.Status == 1
		vo.FailCount = ar.FailCount
		vo.CloseReason = ar.CloseReason
		if ar.NextRenewTime != nil {
			t := ar.NextRenewTime.Format(timeFormat)
			vo.NextRenewTime = &t
		}
	} else {
		vo.PayMethod = "balance"
		vo.Enabled = false
		vo.FailCount = 0
	}
	return vo, nil
}

func (s *OrderService) GetPage(ctx context.Context, q *query.OrderPageQuery) (*vo.PageResult[vo.OrderPageVO], error) {
	list, total, err := s.orderRepo.FindPage(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询订单列表失败", err)
	}
	vos := make([]vo.OrderPageVO, 0, len(list))
	for _, o := range list {
		v := vo.OrderPageVO{
			MyOrderVO:      toMyOrderVO(&o.SysOrder),
			UserID:         o.UserID,
			Username:       o.Username,
			OriginalPrice:  o.OriginalPrice,
			DiscountAmount: o.DiscountAmount,
			CouponAmount:   o.CouponAmount,
		}
		vos = append(vos, v)
	}
	return &vo.PageResult[vo.OrderPageVO]{List: vos, Total: total}, nil
}

func (s *OrderService) ListRefunds(ctx context.Context, q *query.RefundPageQuery) (*vo.PageResult[vo.RefundRecordVO], error) {
	list, total, err := s.refundRepo.FindPage(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询退款列表失败", err)
	}
	vos := make([]vo.RefundRecordVO, 0, len(list))
	for _, r := range list {
		vos = append(vos, *s.toRefundRecordVO(ctx, &r.SysRefundRecord, r.OrderNo, r.Username))
	}
	return &vo.PageResult[vo.RefundRecordVO]{List: vos, Total: total}, nil
}

func (s *OrderService) ApproveRefund(ctx context.Context, auditorID, refundID int64, form *bo.RefundAuditForm) error {
	rr, err := s.refundRepo.FindByID(ctx, refundID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询退款记录失败", err)
	}
	if rr == nil {
		return common.NewBizError(common.REFUND_NOT_FOUND, "退款记录不存在")
	}
	if rr.Status != 1 {
		return common.NewBizError(common.ORDER_STATUS_INVALID, "退款状态不允许此操作")
	}

	now := time.Now()
	return s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txRefundRepo := orderrepo.NewRefundRecordRepository(tx)
		txOrderRepo := orderrepo.NewOrderRepository(tx)

		if err := txRefundRepo.Update(ctx, refundID, map[string]interface{}{
			"status":        2,
			"audit_time":    now,
			"auditor_id":    auditorID,
			"audit_remark":  form.Remark,
			"refund_time":   now,
		}); err != nil {
			return err
		}
		return txOrderRepo.Update(ctx, rr.OrderID, map[string]interface{}{
			"status": 6,
		})
	})
}

func (s *OrderService) RejectRefund(ctx context.Context, auditorID, refundID int64, form *bo.RefundAuditForm) error {
	rr, err := s.refundRepo.FindByID(ctx, refundID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询退款记录失败", err)
	}
	if rr == nil {
		return common.NewBizError(common.REFUND_NOT_FOUND, "退款记录不存在")
	}
	if rr.Status != 1 {
		return common.NewBizError(common.ORDER_STATUS_INVALID, "退款状态不允许此操作")
	}

	now := time.Now()
	return s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txRefundRepo := orderrepo.NewRefundRecordRepository(tx)
		txOrderRepo := orderrepo.NewOrderRepository(tx)

		if err := txRefundRepo.Update(ctx, refundID, map[string]interface{}{
			"status":       3,
			"audit_time":   now,
			"auditor_id":   auditorID,
			"audit_remark": form.Remark,
		}); err != nil {
			return err
		}
		return txOrderRepo.Update(ctx, rr.OrderID, map[string]interface{}{
			"status": 2,
		})
	})
}

func (s *OrderService) GetStats(ctx context.Context, startTime, endTime string) (*vo.OrderStatsVO, error) {
	stats := &vo.OrderStatsVO{
		StatusDistribution:    make(map[string]int64),
		PayMethodDistribution: make(map[string]int64),
		PackageDistribution:   make([]vo.OrderPackageStatItem, 0),
		DailyStats:            make([]vo.OrderDailyStatItem, 0),
	}

	totalOrders, err := s.orderRepo.CountTotalOrders(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询订单总数失败", err)
	}
	stats.TotalOrders = totalOrders

	revenue, err := s.orderRepo.SumRevenue(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询收入失败", err)
	}
	stats.TotalRevenue = revenue

	refund, err := s.orderRepo.SumRefund(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询退款失败", err)
	}
	stats.TotalRefund = refund

	if revenue > 0 {
		stats.RefundRate = refund * 100 / revenue
	}

	statuses := []struct {
		code int8
		name string
	}{
		{1, "pending"}, {2, "paid"}, {3, "completed"},
		{4, "cancelled"}, {5, "refunding"}, {6, "refunded"},
	}
	for _, st := range statuses {
		count, _ := s.orderRepo.CountByStatus(ctx, st.code)
		stats.StatusDistribution[st.name] = count
	}

	payMethods := []string{"wechat", "alipay", "balance", "combined"}
	for _, pm := range payMethods {
		count, _ := s.orderRepo.CountByPayMethod(ctx, pm)
		stats.PayMethodDistribution[pm] = count
	}

	pkgRows, err := s.orderRepo.GetPackageDistribution(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询套餐分布失败", err)
	}
	for _, row := range pkgRows {
		stats.PackageDistribution = append(stats.PackageDistribution, vo.OrderPackageStatItem{
			PackageID:   row.PackageID,
			PackageName: row.PackageName,
			Count:       row.Count,
			Revenue:     row.Revenue,
		})
	}

	dailyRows, err := s.orderRepo.GetDailyStats(ctx, startTime, endTime)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询日统计失败", err)
	}
	for _, row := range dailyRows {
		stats.DailyStats = append(stats.DailyStats, vo.OrderDailyStatItem{
			Date:    row.Date,
			Count:   row.Count,
			Revenue: row.Revenue,
		})
	}

	return stats, nil
}

func (s *OrderService) CancelExpiredOrders(ctx context.Context) error {
	list, err := s.orderRepo.FindPendingExpired(ctx, time.Now())
	if err != nil {
		return err
	}
	for _, o := range list {
		_ = s.orderRepo.Update(ctx, o.ID, map[string]interface{}{
			"status":        4,
			"cancel_reason": "支付超时自动取消",
		})
	}
	return nil
}

func (s *OrderService) ProcessAutoRenewals(ctx context.Context) error {
	return nil
}

func (s *OrderService) ExpireUserCoupons(ctx context.Context) error {
	list, err := s.userCouponRepo.FindExpired(ctx, time.Now())
	if err != nil {
		return err
	}
	if len(list) == 0 {
		return nil
	}
	ids := make([]int64, 0, len(list))
	for _, uc := range list {
		ids = append(ids, uc.ID)
	}
	return s.userCouponRepo.BatchMarkExpired(ctx, ids)
}

func (s *OrderService) toRefundRecordVO(ctx context.Context, r *model.SysRefundRecord, orderNo, username string) *vo.RefundRecordVO {
	v := &vo.RefundRecordVO{
		ID:              r.ID,
		RefundNo:        r.RefundNo,
		OrderID:         r.OrderID,
		OrderNo:         orderNo,
		UserID:          r.UserID,
		Username:        username,
		RefundAmount:    r.RefundAmount,
		Reason:          r.Reason,
		UsedQuota:       r.UsedQuota,
		Status:          orderrepo.RefundStatusToString(r.Status),
		ChannelRefundNo: r.ChannelRefundNo,
		ApplyTime:       r.ApplyTime.Format(timeFormat),
		AuditRemark:     r.AuditRemark,
		ErrorMessage:    r.ErrorMessage,
	}
	if r.Channel != nil {
		v.Channel = r.Channel
	}
	if r.AuditTime != nil {
		t := r.AuditTime.Format(timeFormat)
		v.AuditTime = &t
	}
	if r.AuditorID != nil {
		v.AuditorID = r.AuditorID
	}
	if r.RefundTime != nil {
		t := r.RefundTime.Format(timeFormat)
		v.RefundTime = &t
	}
	return v
}

func (s *OrderService) findUsernameByUserID(ctx context.Context, userID int64) (string, error) {
	var username string
	err := s.db.WithContext(ctx).
		Table("sys_user").
		Where("id = ? AND deleted = 0", userID).
		Select("username").
		Scan(&username).Error
	if err != nil {
		return "", err
	}
	return username, nil
}

func toMyOrderVO(o *model.SysOrder) vo.MyOrderVO {
	v := vo.MyOrderVO{
		ID:            o.ID,
		OrderNo:       o.OrderNo,
		PackageName:   o.PackageName,
		PackageLevel:  o.PackageLevel,
		PayableAmount: o.PayableAmount,
		PaidAmount:    o.PaidAmount,
		PayMethod:     o.PayMethod,
		Status:        orderrepo.OrderStatusToString(o.Status),
		CreateTime:    o.CreatedAt.Format(timeFormat),
	}
	if o.PaidTime != nil {
		t := o.PaidTime.Format(timeFormat)
		v.PaidTime = &t
	}
	if o.PackageExpireTime != nil {
		t := o.PackageExpireTime.Format(timeFormat)
		v.PackageExpireTime = &t
	}
	return v
}

func generateOrderNo() string {
	return fmt.Sprintf("OD%s%06d", time.Now().Format("20060102150405"), rand.Intn(1000000))
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

var _ IOrderService = (*OrderService)(nil)

package order

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	memberrepo "github.com/earthyzinc/dehaze-go/internal/repository/member"
	orderrepo "github.com/earthyzinc/dehaze-go/internal/repository/order"
	pkgsalerepo "github.com/earthyzinc/dehaze-go/internal/repository/pkgsale"
	userrepo "github.com/earthyzinc/dehaze-go/internal/repository/user"
	auditlogservice "github.com/earthyzinc/dehaze-go/internal/service/audit_log"
	paymentsvc "github.com/earthyzinc/dehaze-go/internal/service/payment"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/database"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
	"gorm.io/gorm"
)

const (
	orderExpireMinutes = 30
	timeFormat         = "2006-01-02 15:04:05"

	orderDetailCacheTTL    = 10 * time.Minute
	orderCreateLockTTL     = 5 * time.Second
	paymentCallbackLockTTL = 30 * time.Second
	autoRenewMaxFailCount  = 3
)

type OrderService struct {
	db                     *gorm.DB
	orderRepo              orderrepo.IOrderRepository
	paymentRepo            orderrepo.IPaymentRecordRepository
	refundRepo             orderrepo.IRefundRecordRepository
	autoRenewRepo          orderrepo.IAutoRenewRepository
	packageRepo            pkgsalerepo.IPackageRepository
	couponRepo             pkgsalerepo.ICouponRepository
	userCouponRepo         pkgsalerepo.IUserCouponRepository
	userRepo               userrepo.IUserRepository
	memberRepo             memberrepo.IMemberRepository
	benefitRepo            memberrepo.IMemberBenefitRepository
	paymentSvc             paymentsvc.IPaymentChannelService
	cache                  types.ICache
	auditLogSvc            *auditlogservice.AuditLogService
	memberCacheInvalidator MemberCacheInvalidator
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
	userRepo userrepo.IUserRepository,
	memberRepo memberrepo.IMemberRepository,
	benefitRepo memberrepo.IMemberBenefitRepository,
	paymentSvc paymentsvc.IPaymentChannelService,
	cache types.ICache,
	auditLogSvc *auditlogservice.AuditLogService,
	memberCacheInvalidator MemberCacheInvalidator,
) *OrderService {
	return &OrderService{
		db:                     db,
		orderRepo:              orderRepo,
		paymentRepo:            paymentRepo,
		refundRepo:             refundRepo,
		autoRenewRepo:          autoRenewRepo,
		packageRepo:            packageRepo,
		couponRepo:             couponRepo,
		userCouponRepo:         userCouponRepo,
		userRepo:               userRepo,
		memberRepo:             memberRepo,
		benefitRepo:            benefitRepo,
		paymentSvc:             paymentSvc,
		cache:                  cache,
		auditLogSvc:            auditLogSvc,
		memberCacheInvalidator: memberCacheInvalidator,
	}
}

// validPayMethods 支付方式白名单（python order_service.PAY_METHODS 同口径）
var validPayMethods = map[string]bool{"wechat": true, "alipay": true, "balance": true, "combined": true}

func (s *OrderService) Create(ctx context.Context, userID int64, form *bo.OrderCreateForm) (*vo.PayResult, error) {
	if !validPayMethods[form.PayMethod] {
		return nil, common.NewBizError(common.PARAM_ERROR, "不支持的支付方式")
	}

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

	lockKey := OrderCreateLockKey(userID, form.PackageID)
	if s.cache != nil {
		token, ok, _ := s.cache.Lock(ctx, lockKey, orderCreateLockTTL)
		if !ok {
			return nil, common.NewBizError(common.DUPLICATE_ORDER, "请勿短时间内重复下单")
		}
		defer func() { _, _ = s.cache.Unlock(ctx, lockKey, token) }()
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
		PackageType:    p.PackageType,
		OriginalPrice:  p.OriginalPrice,
		DiscountAmount: discountAmount,
		CouponID:       userCouponID,
		CouponAmount:   couponAmount,
		PayableAmount:  payableAmount,
		PaidAmount:     0,
		PayMethod:      &form.PayMethod,
		Status:         1,
		ExpireTime:     expireTime,
		IsAutoRenew:    0,
	}
	// 冗余字段按商品类型落库：vip 存等级/周期，credit 存可得积分（python create 同口径）
	if p.PackageType == "credit" {
		order.CreditAmount = creditAmountPtr(p.CreditAmount)
	} else {
		order.PackageLevel = p.LevelCode.String
		order.PeriodDays = periodDaysPtr(p.PeriodDays)
	}

	payMethod := form.PayMethod

	err = s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txOrderRepo := orderrepo.NewOrderRepository(tx)
		txUserCouponRepo := pkgsalerepo.NewUserCouponRepository(tx)

		if err := txOrderRepo.Create(ctx, order); err != nil {
			return err
		}

		if userCouponID != nil {
			if err := txUserCouponRepo.Update(ctx, *userCouponID, map[string]interface{}{
				"status": 4,
			}); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建订单失败", err)
	}

	result := &vo.PayResult{
		OrderNo:   orderNo,
		PayMethod: payMethod,
		Paid:      false,
	}
	if s.auditLogSvc != nil {
		s.auditLogSvc.RecordAuditAsync(ctx, database.GetUserID(ctx), "order", orderNo, "create", "order", nil, form, database.GetIP(ctx), database.GetUserAgent(ctx))
	}
	return result, nil
}

func (s *OrderService) completePaymentInTx(ctx context.Context, tx *gorm.DB, order *model.SysOrder, channel string) error {
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
		Channel:      channel,
		Amount:       order.PayableAmount,
		Status:       2,
		CallbackTime: &now,
	}
	if err := txPaymentRepo.Create(ctx, payment); err != nil {
		return err
	}

	// 积分卡支付即完成（status 3）且无履约有效期；会员卡支付后进入已支付(2)
	status := int8(2)
	if order.PackageType == "credit" {
		status = 3
	}

	updates := map[string]interface{}{
		"status":      status,
		"paid_amount": order.PayableAmount,
		"paid_time":   now,
		"pay_method":  channel,
	}

	var packageExpireTime time.Time
	if order.PackageType != "credit" {
		effectiveTime := now
		packageExpireTime = now.AddDate(0, 0, orderPeriodDays(order))

		member, _ := txMemberRepo.FindByUserID(ctx, order.UserID)
		if member != nil && member.ExpireTime != nil && member.ExpireTime.After(now) {
			effectiveTime = *member.ExpireTime
			packageExpireTime = effectiveTime.AddDate(0, 0, orderPeriodDays(order))
		}
		updates["effective_time"] = effectiveTime
		updates["package_expire_time"] = packageExpireTime
	}

	if err := txOrderRepo.Update(ctx, order.ID, updates); err != nil {
		return err
	}

	if order.CouponID != nil && *order.CouponID > 0 {
		uc, _ := txUserCouponRepo.FindByID(ctx, *order.CouponID)
		if uc != nil {
			if err := txUserCouponRepo.Update(ctx, uc.ID, map[string]interface{}{
				"status":        2,
				"used_time":     now,
				"used_order_id": order.ID,
			}); err != nil {
				return err
			}
			_ = txCouponRepo.IncrementUsedQty(ctx, uc.CouponID)
		}
	}

	pkg, _ := s.packageRepo.FindByID(ctx, order.PackageID)
	if err := s.updateMemberAfterPaymentInTx(ctx, txMemberRepo, order, order.PayableAmount, &packageExpireTime, pkg); err != nil {
		return err
	}
	s.invalidateMemberCacheAfterPayment(ctx, order.UserID, order.PackageLevel)
	return nil
}

// updateMemberAfterPaymentInTx 支付履约（python member_service.on_order_paid 同口径）：
// 所有商品按实付 1:1 累积成长值与累计消费；会员卡升级/续期/刷新权益，积分卡不动等级权益。
func (s *OrderService) updateMemberAfterPaymentInTx(ctx context.Context, txMemberRepo memberrepo.IMemberRepository, order *model.SysOrder, amount int64, expireTime *time.Time, pkg *model.SysPackage) error {
	userID := order.UserID
	levelCode := order.PackageLevel

	member, _ := txMemberRepo.FindByUserID(ctx, userID)
	if member == nil {
		return common.NewBizError(common.MEMBER_NOT_FOUND, "会员不存在")
	}

	// 成长值累积：实付金额 1:1（1 元 = 100 分 = 100 成长值）
	updates := map[string]interface{}{
		"growth_value":      member.GrowthValue + amount,
		"total_consumption": member.TotalConsumption + amount,
	}

	if order.PackageType == "vip" && levelCode != "" {
		benefit, _ := s.benefitRepo.FindByLevelCode(ctx, levelCode)
		dehazeQuota := 0
		evaluateQuota := 0
		if benefit != nil {
			dehazeQuota = benefit.MonthlyDehazeQuota
			evaluateQuota = benefit.MonthlyEvaluateQuota
		}
		if pkg != nil && pkg.BenefitOverrides.Valid {
			var overrides map[string]int
			if err := json.Unmarshal([]byte(pkg.BenefitOverrides.String), &overrides); err == nil {
				if v, ok := overrides["monthlyDehazeQuota"]; ok {
					dehazeQuota = v
				}
				if v, ok := overrides["monthlyEvaluateQuota"]; ok {
					evaluateQuota = v
				}
			}
		}

		// 等级只升不降（需求 §3.1.1 会员卡期间享有较高者）
		if memberLevelRank(member.LevelCode) > memberLevelRank(levelCode) {
			levelCode = member.LevelCode
		}
		updates["level_code"] = levelCode
		updates["level_source"] = "purchase"
		updates["expire_time"] = *expireTime
		updates["status"] = 1
		updates["monthly_dehaze_quota"] = dehazeQuota
		updates["monthly_evaluate_quota"] = evaluateQuota
		if member.BecomeMemberTime == nil {
			updates["become_member_time"] = time.Now()
		}
	}
	return txMemberRepo.Update(ctx, userID, updates)
}

func (s *OrderService) invalidateMemberCacheAfterPayment(ctx context.Context, userID int64, levelCode string) {
	if s.memberCacheInvalidator != nil {
		s.memberCacheInvalidator.InvalidateMemberCache(ctx, userID, levelCode)
	}
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
	cacheKey := OrderDetailKey(orderNo)
	if s.cache != nil {
		if cached, err := s.cache.Get(ctx, cacheKey); err == nil && cached != "" {
			var detail vo.OrderDetailVO
			if err := json.Unmarshal([]byte(cached), &detail); err == nil {
				return &detail, nil
			}
		}
	}

	o, err := s.orderRepo.FindByOrderNo(ctx, orderNo)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询订单失败", err)
	}
	if o == nil || o.UserID != database.GetUserID(ctx) {
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
			ID:         p.ID,
			PaymentNo:  p.PaymentNo,
			Channel:    p.Channel,
			Amount:     p.Amount,
			Status:     int(p.Status),
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

	if s.cache != nil {
		if data, err := json.Marshal(detail); err == nil {
			_ = s.cache.Set(ctx, cacheKey, string(data), orderDetailCacheTTL)
		}
	}

	return detail, nil
}

func (s *OrderService) invalidateOrderDetailCache(ctx context.Context, orderNo string) {
	if s.cache == nil {
		return
	}
	_ = s.cache.Delete(ctx, OrderDetailKey(orderNo))
}

func (s *OrderService) Cancel(ctx context.Context, orderNo string, reason string) error {
	o, err := s.orderRepo.FindByOrderNo(ctx, orderNo)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询订单失败", err)
	}
	if o == nil || o.UserID != database.GetUserID(ctx) {
		return common.NewBizError(common.ORDER_NOT_FOUND, "订单不存在")
	}
	if o.Status != 1 {
		return common.NewBizError(common.ORDER_STATUS_INVALID, "订单状态不允许此操作")
	}

	err = s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
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
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "取消订单失败", err)
	}
	s.invalidateOrderDetailCache(ctx, orderNo)
	if s.auditLogSvc != nil {
		s.auditLogSvc.RecordAuditAsync(ctx, database.GetUserID(ctx), "order", orderNo, "cancel", "order", nil, map[string]interface{}{"reason": reason}, database.GetIP(ctx), database.GetUserAgent(ctx))
	}
	return nil
}

func (s *OrderService) Pay(ctx context.Context, orderNo string, req *bo.PayRequest) (*vo.PayResult, error) {
	o, err := s.orderRepo.FindByOrderNo(ctx, orderNo)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询订单失败", err)
	}
	if o == nil || o.UserID != database.GetUserID(ctx) {
		return nil, common.NewBizError(common.ORDER_NOT_FOUND, "订单不存在")
	}
	if o.Status != 1 {
		if o.Status == 2 {
			return nil, common.NewBizError(common.ORDER_ALREADY_PAID, "订单已支付")
		}
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
		// 余额支付：校验并扣减平台余额（乐观锁 CAS），不足拒绝（A053B）
		accountRepo := orderrepo.NewBalanceAccountRepository(s.db)
		account, err := accountRepo.GetOrCreate(ctx, o.UserID)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询余额账户失败", err)
		}
		if account.Balance < o.PayableAmount {
			return nil, common.NewBizError(common.BALANCE_INSUFFICIENT, "余额不足")
		}
		ok, err := accountRepo.AdjustBalance(ctx, o.UserID, -o.PayableAmount, account.Version)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "扣减余额失败", err)
		}
		if !ok {
			return nil, common.NewBizError(common.BALANCE_INSUFFICIENT, "余额不足")
		}
		if err := orderrepo.NewBalanceLogRepository(s.db).Create(ctx, &model.SysBalanceLog{
			UserID:       o.UserID,
			ChangeType:   "consume",
			Amount:       -o.PayableAmount,
			BalanceAfter: account.Balance - o.PayableAmount,
			RelatedID:    &o.ID,
		}); err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "写入余额流水失败", err)
		}

		err = s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
			return s.completePaymentInTx(ctx, tx, o, req.PayMethod)
		})
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "支付失败", err)
		}
		result.Paid = true
		_ = s.packageRepo.IncrementSalesCount(ctx, o.PackageID, 1)
		s.invalidateOrderDetailCache(ctx, orderNo)
	} else if s.paymentSvc != nil {
		payResult, payErr := s.paymentSvc.CreateOrder(ctx, &paymentsvc.UnifiedOrderRequest{
			OrderNo:     orderNo,
			Amount:      o.PayableAmount,
			Description: o.PackageName,
			PayMethod:   req.PayMethod,
		})
		if payErr != nil {
			return nil, common.WrapBizError(common.CALL_THIRD_PARTY_SERVICE_ERROR, "调用支付渠道下单失败", payErr)
		}
		result.Paid = false
		result.PayURL = payResult.PayURL
		result.QRCode = payResult.QRCode
	} else {
		result.Paid = false
	}

	return result, nil
}

// calcRefundAmount 按商品类型折算退款金额（python _calc_refund_amount 同口径）。
// 返回 (refundAmount, usedDays, usedCredits)：merchant 全额退；vip 按天折算；
// credit 按积分消耗折算（go 侧 AI 积分域未落地，暂按未消耗计，待 billing 域补齐后接入积分余额）。
func calcRefundAmount(o *model.SysOrder, reasonType string) (int64, *int, *int64) {
	if reasonType == "merchant" {
		return o.PaidAmount, nil, nil
	}
	paidTime := time.Now()
	if o.PaidTime != nil {
		paidTime = *o.PaidTime
	}

	if o.PackageType == "credit" {
		usedCredits := int64(0)
		return o.PaidAmount, nil, &usedCredits
	}

	// vip 按天折算
	periodDays := 0
	if o.PeriodDays != nil {
		periodDays = *o.PeriodDays
	}
	if periodDays <= 0 {
		return 0, nil, nil
	}
	usedDays := int(math.Ceil(time.Since(paidTime).Hours() / 24))
	if usedDays < 1 {
		usedDays = 1
	}
	remaining := periodDays - usedDays
	if remaining <= 0 {
		return 0, &usedDays, nil
	}
	refundAmount := o.PaidAmount * int64(remaining) / int64(periodDays)
	return refundAmount, &usedDays, nil
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
		return common.NewBizError(common.REFUND_ALREADY_EXISTS, "该订单已存在退款申请")
	}

	// 契约（后端实现 §2.5）：customReason 非空时以 reasonType + ":" + customReason 拼接写入
	reason := form.ReasonType
	if form.CustomReason != "" {
		reason = form.ReasonType + ":" + form.CustomReason
	}

	refundAmount, usedDays, usedCredits := calcRefundAmount(o, form.ReasonType)

	refundNo := fmt.Sprintf("RF%s%06d", time.Now().Format("20060102150405"), rand.Intn(1000000))
	refund := &model.SysRefundRecord{
		RefundNo:     refundNo,
		OrderID:      o.ID,
		UserID:       userID,
		RefundAmount: refundAmount,
		ReasonType:   form.ReasonType,
		Reason:       reason,
		UsedDays:     usedDays,
		UsedCredits:  usedCredits,
		Status:       1,
		Channel:      o.PayMethod,
		ApplyTime:    time.Now(),
	}

	err = s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txRefundRepo := orderrepo.NewRefundRecordRepository(tx)
		txOrderRepo := orderrepo.NewOrderRepository(tx)

		if err := txRefundRepo.Create(ctx, refund); err != nil {
			return err
		}
		return txOrderRepo.Update(ctx, o.ID, map[string]interface{}{
			"status": 5,
		})
	})
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "申请退款失败", err)
	}
	s.invalidateOrderDetailCache(ctx, orderNo)
	if s.auditLogSvc != nil {
		s.auditLogSvc.RecordAuditAsync(ctx, userID, "order", orderNo, "refund_apply", "order", nil, form, database.GetIP(ctx), database.GetUserAgent(ctx))
	}
	return nil
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

	o, err := s.orderRepo.FindByID(ctx, rr.OrderID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询订单失败", err)
	}
	if o == nil {
		return common.NewBizError(common.ORDER_NOT_FOUND, "订单不存在")
	}

	channel := ""
	if rr.Channel != nil {
		channel = *rr.Channel
	}
	if channel == "" {
		if o.PayMethod != nil {
			channel = *o.PayMethod
		} else {
			channel = "balance"
		}
	}

	if s.paymentSvc != nil {
		refundResult, refundErr := s.paymentSvc.Refund(ctx, &paymentsvc.RefundRequest{
			OrderNo:   o.OrderNo,
			PaymentNo: rr.RefundNo,
			Channel:   channel,
			Amount:    rr.RefundAmount,
			Reason:    rr.Reason,
		})
		if refundErr != nil {
			logger.Error("调用支付渠道退款失败", zap.String("orderNo", o.OrderNo), zap.Error(refundErr))
			return common.WrapBizError(common.CALL_THIRD_PARTY_SERVICE_ERROR, "调用支付渠道退款失败", refundErr)
		}
		if !refundResult.Success {
			errMsg := refundResult.ErrorMessage
			if errMsg == "" {
				errMsg = "渠道退款失败"
			}
			_ = s.refundRepo.Update(ctx, refundID, map[string]interface{}{
				"status":        3,
				"audit_time":    time.Now(),
				"auditor_id":    auditorID,
				"audit_remark":  form.Remark,
				"error_message": errMsg,
			})
			restoreStatus := int8(2)
			if o.PackageExpireTime != nil && o.PackageExpireTime.After(time.Now()) {
				restoreStatus = 3
			}
			_ = s.orderRepo.Update(ctx, o.ID, map[string]interface{}{
				"status": restoreStatus,
			})
			return common.NewBizError(common.CALL_THIRD_PARTY_SERVICE_ERROR, errMsg)
		}
	}

	now := time.Now()
	err = s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		txRefundRepo := orderrepo.NewRefundRecordRepository(tx)
		txOrderRepo := orderrepo.NewOrderRepository(tx)

		updates := map[string]interface{}{
			"status":       2,
			"audit_time":   now,
			"auditor_id":   auditorID,
			"audit_remark": form.Remark,
			"refund_time":  now,
		}
		if s.paymentSvc != nil {
			updates["channel"] = channel
		}
		if err := txRefundRepo.Update(ctx, refundID, updates); err != nil {
			return err
		}
		return txOrderRepo.Update(ctx, rr.OrderID, map[string]interface{}{
			"status": 6,
		})
	})
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "审核退款失败", err)
	}
	s.invalidateOrderDetailCache(ctx, o.OrderNo)
	if s.auditLogSvc != nil {
		s.auditLogSvc.RecordAuditAsync(ctx, auditorID, "order", refundID, "refund_approve", "order", nil, form, database.GetIP(ctx), database.GetUserAgent(ctx))
	}
	return nil
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

	o, err := s.orderRepo.FindByID(ctx, rr.OrderID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询订单失败", err)
	}
	if o == nil {
		return common.NewBizError(common.ORDER_NOT_FOUND, "订单不存在")
	}

	restoreStatus := int8(2)
	if o.PackageExpireTime != nil && o.PackageExpireTime.After(time.Now()) {
		restoreStatus = 3
	}

	now := time.Now()
	err = s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
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
			"status": restoreStatus,
		})
	})
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "拒绝退款失败", err)
	}
	s.invalidateOrderDetailCache(ctx, o.OrderNo)
	if s.auditLogSvc != nil {
		s.auditLogSvc.RecordAuditAsync(ctx, auditorID, "order", refundID, "refund_reject", "order", nil, form, database.GetIP(ctx), database.GetUserAgent(ctx))
	}
	return nil
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
		stats.RefundRate = float64(refund) / float64(revenue)
	}

	statuses := []struct {
		code int8
		name string
	}{
		{1, "pending"}, {2, "paid"}, {3, "completed"},
		{4, "cancelled"}, {5, "refunding"}, {6, "refunded"},
	}
	for _, st := range statuses {
		count, _ := s.orderRepo.CountByStatus(ctx, st.code, startTime, endTime)
		stats.StatusDistribution[st.name] = count
	}

	payMethods := []string{"wechat", "alipay", "balance", "combined"}
	for _, pm := range payMethods {
		count, _ := s.orderRepo.CountByPayMethod(ctx, pm, startTime, endTime)
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
		if o.PayMethod != nil && (*o.PayMethod == "wechat" || *o.PayMethod == "alipay") && s.paymentSvc != nil {
			if closeErr := s.paymentSvc.ChannelCloseOrder(ctx, *o.PayMethod, o.OrderNo); closeErr != nil {
				logger.Warn("超时关单失败", zap.String("orderNo", o.OrderNo), zap.Error(closeErr))
			}
		}
		if err := s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
			if err := tx.Model(&model.SysOrder{}).Where("id = ?", o.ID).Updates(map[string]interface{}{
				"status":        4,
				"cancel_reason": "支付超时自动取消",
			}).Error; err != nil {
				return err
			}
			if o.CouponID != nil && *o.CouponID > 0 {
				if err := tx.Model(&model.SysUserCoupon{}).Where("id = ?", *o.CouponID).Updates(map[string]interface{}{
					"status": 1,
				}).Error; err != nil {
					return err
				}
			}
			return nil
		}); err != nil {
			logger.Error("取消超时订单失败", zap.String("orderNo", o.OrderNo), zap.Error(err))
		}
	}
	return nil
}

func (s *OrderService) CompleteExpiredOrders(ctx context.Context) error {
	list, err := s.orderRepo.FindPaidExpired(ctx, time.Now())
	if err != nil {
		return err
	}
	for _, o := range list {
		_ = s.orderRepo.Update(ctx, o.ID, map[string]interface{}{
			"status": 3,
		})
		s.invalidateOrderDetailCache(ctx, o.OrderNo)
	}
	return nil
}

func (s *OrderService) ProcessAutoRenewals(ctx context.Context) error {
	dueList, err := s.autoRenewRepo.FindDueRenewals(ctx, time.Now())
	if err != nil {
		return fmt.Errorf("查询到期自动续费记录失败: %w", err)
	}

	for _, ar := range dueList {
		if err := s.processSingleAutoRenewal(ctx, &ar); err != nil {
			logger.Error("处理自动续费失败",
				zap.Int64("userId", ar.UserID),
				zap.Int64("packageId", ar.PackageID),
				zap.Error(err))
		}
	}
	return nil
}

func (s *OrderService) processSingleAutoRenewal(ctx context.Context, ar *model.SysAutoRenew) error {
	p, err := s.packageRepo.FindByID(ctx, ar.PackageID)
	if err != nil {
		return err
	}
	if p == nil || p.Status != 1 {
		_ = s.autoRenewRepo.Update(ctx, ar.ID, map[string]interface{}{
			"status":       0,
			"close_reason": "套餐已下架",
		})
		return nil
	}

	member, _ := s.memberRepo.FindByUserID(ctx, ar.UserID)
	if member == nil || member.Status != 1 {
		_ = s.autoRenewRepo.Update(ctx, ar.ID, map[string]interface{}{
			"status":       0,
			"close_reason": "会员不存在或已冻结",
		})
		return nil
	}

	orderNo := generateOrderNo()
	now := time.Now()
	expireTime := now.Add(orderExpireMinutes * time.Minute)

	payableAmount := int64(float64(p.SalePrice) * 0.95)

	order := &model.SysOrder{
		OrderNo:        orderNo,
		UserID:         ar.UserID,
		PackageID:      p.ID,
		PackageName:    p.Name,
		PackageType:    p.PackageType,
		PackageLevel:   p.LevelCode.String,
		PeriodDays:     periodDaysPtr(p.PeriodDays),
		OriginalPrice:  p.OriginalPrice,
		DiscountAmount: p.SalePrice - payableAmount,
		PayableAmount:  payableAmount,
		Status:         1,
		ExpireTime:     expireTime,
		IsAutoRenew:    1,
	}

	if err := s.orderRepo.Create(ctx, order); err != nil {
		return err
	}

	if ar.PayMethod == "balance" {
		if err := s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
			return s.completePaymentInTx(ctx, tx, order, ar.PayMethod)
		}); err != nil {
			return err
		}
		_ = s.packageRepo.IncrementSalesCount(ctx, p.ID, 1)
	} else if s.paymentSvc != nil {
		_, payErr := s.paymentSvc.CreateOrder(ctx, &paymentsvc.UnifiedOrderRequest{
			OrderNo:     orderNo,
			Amount:      payableAmount,
			Description: p.Name + "(自动续费)",
			PayMethod:   ar.PayMethod,
		})
		if payErr != nil {
			newFailCount := ar.FailCount + 1
			updates := map[string]interface{}{
				"fail_count": newFailCount,
			}
			if newFailCount >= autoRenewMaxFailCount {
				updates["status"] = 0
				updates["close_reason"] = "连续扣款失败超过限制"
			} else {
				nextRenew := time.Now().Add(2 * time.Hour)
				updates["next_renew_time"] = nextRenew
			}
			_ = s.autoRenewRepo.Update(ctx, ar.ID, updates)
			return payErr
		}
	}

	nextRenew := now.AddDate(0, 0, int(p.PeriodDays.Int64))
	return s.autoRenewRepo.Update(ctx, ar.ID, map[string]interface{}{
		"fail_count":          0,
		"next_renew_time":     nextRenew,
		"last_renew_order_id": order.ID,
	})
}

func (s *OrderService) HandlePaymentCallback(ctx context.Context, channel, orderNo, channelNo string, amount int64, success bool, rawContent string) error {
	o, err := s.orderRepo.FindByOrderNo(ctx, orderNo)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询订单失败", err)
	}
	if o == nil {
		return common.NewBizError(common.ORDER_NOT_FOUND, "订单不存在")
	}
	if o.Status != 1 {
		return nil
	}

	if amount > 0 && amount != o.PayableAmount {
		logger.Error("支付回调金额不一致", zap.String("orderNo", orderNo), zap.Int64("expected", o.PayableAmount), zap.Int64("actual", amount))
		return common.NewBizError(common.PAYMENT_AMOUNT_MISMATCH, "支付金额与订单金额不一致")
	}

	lockKey := fmt.Sprintf("payment:callback:lock:%s", orderNo)
	if s.cache != nil {
		token, ok, _ := s.cache.Lock(ctx, lockKey, paymentCallbackLockTTL)
		if !ok {
			return common.NewBizError(common.REPEAT_SUBMIT_ERROR, "正在处理支付回调，请勿重复提交")
		}
		defer func() { _, _ = s.cache.Unlock(ctx, lockKey, token) }()
	}

	now := time.Now()
	payment := &model.SysPaymentRecord{
		OrderID:         o.ID,
		UserID:          o.UserID,
		PaymentNo:       channelNo,
		Channel:         channel,
		Amount:          amount,
		Status:          2,
		CallbackTime:    &now,
		CallbackContent: rawContent,
	}
	if success {
		err = s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
			txPaymentRepo := orderrepo.NewPaymentRecordRepository(tx)
			if err := txPaymentRepo.Create(ctx, payment); err != nil {
				return err
			}
			txOrderRepo := orderrepo.NewOrderRepository(tx)
			txMemberRepo := memberrepo.NewMemberRepository(tx)
			txUserCouponRepo := pkgsalerepo.NewUserCouponRepository(tx)
			txCouponRepo := pkgsalerepo.NewCouponRepository(tx)

			effectiveTime := now
			packageExpireTime := now.AddDate(0, 0, orderPeriodDays(o))

			member, _ := txMemberRepo.FindByUserID(ctx, o.UserID)
			if member != nil && member.ExpireTime != nil && member.ExpireTime.After(now) {
				effectiveTime = *member.ExpireTime
				packageExpireTime = effectiveTime.AddDate(0, 0, orderPeriodDays(o))
			}

			if err := txOrderRepo.Update(ctx, o.ID, map[string]interface{}{
				"status":              2,
				"paid_amount":         amount,
				"paid_time":           now,
				"pay_method":          channel,
				"effective_time":      effectiveTime,
				"package_expire_time": packageExpireTime,
			}); err != nil {
				return err
			}
			if o.CouponID != nil && *o.CouponID > 0 {
				uc, _ := txUserCouponRepo.FindByID(ctx, *o.CouponID)
				if uc != nil {
					if err := txUserCouponRepo.Update(ctx, uc.ID, map[string]interface{}{
						"status":        2,
						"used_time":     now,
						"used_order_id": o.ID,
					}); err != nil {
						return err
					}
					_ = txCouponRepo.IncrementUsedQty(ctx, uc.CouponID)
				}
			}
			pkg, _ := s.packageRepo.FindByID(ctx, o.PackageID)
			return s.updateMemberAfterPaymentInTx(ctx, txMemberRepo, o, amount, &packageExpireTime, pkg)
		})
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "支付回调处理失败", err)
		}
		_ = s.packageRepo.IncrementSalesCount(ctx, o.PackageID, 1)
		s.invalidateMemberCacheAfterPayment(ctx, o.UserID, o.PackageLevel)
	} else {
		// 支付失败回调：流水记录一次失败尝试，无事务中段失败回滚语义，独立落库
		payment.Status = 3
		if err := s.paymentRepo.Create(ctx, payment); err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "创建支付记录失败", err)
		}
	}

	s.invalidateOrderDetailCache(ctx, orderNo)
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

func (s *OrderService) RetryFailedRefunds(ctx context.Context) error {
	maxRetryCount := 3
	list, err := s.refundRepo.FindFailedRetryable(ctx, maxRetryCount)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询失败退款记录失败", err)
	}
	if len(list) == 0 {
		return nil
	}

	successCount := 0
	finalFailCount := 0
	for _, rr := range list {
		o, err := s.orderRepo.FindByID(ctx, rr.OrderID)
		if err != nil {
			logger.Error("查询订单失败", zap.Int64("orderId", rr.OrderID), zap.Error(err))
			continue
		}
		if o == nil {
			logger.Warn("退款重试跳过: 退款记录对应订单不存在", zap.Int64("refundId", rr.ID))
			continue
		}

		newRetryCount := int(rr.RetryCount) + 1
		channel := "balance"
		if rr.Channel != nil && *rr.Channel != "" {
			channel = *rr.Channel
		} else if o.PayMethod != nil && *o.PayMethod != "" {
			channel = *o.PayMethod
		}

		refundOk := true
		errorMessage := ""
		if channel != "balance" && s.paymentSvc != nil {
			refundResult, refundErr := s.paymentSvc.Refund(ctx, &paymentsvc.RefundRequest{
				OrderNo:   o.OrderNo,
				PaymentNo: rr.RefundNo,
				Channel:   channel,
				Amount:    rr.RefundAmount,
				Reason:    rr.Reason,
			})
			if refundErr != nil {
				logger.Error("渠道退款重试失败", zap.String("orderNo", o.OrderNo), zap.Int64("refundId", rr.ID), zap.Error(refundErr))
				refundOk = false
				errorMessage = refundErr.Error()
			} else if !refundResult.Success {
				refundOk = false
				errorMessage = refundResult.ErrorMessage
				if errorMessage == "" {
					errorMessage = "渠道退款失败"
				}
			}
		}

		now := time.Now()
		if refundOk {
			updates := map[string]interface{}{
				"status":        2,
				"refund_time":   now,
				"retry_count":   newRetryCount,
				"error_message": "",
			}
			if err := s.refundRepo.Update(ctx, rr.ID, updates); err != nil {
				logger.Error("更新退款记录失败", zap.Int64("refundId", rr.ID), zap.Error(err))
				continue
			}
			if err := s.orderRepo.Update(ctx, o.ID, map[string]interface{}{"status": 6}); err != nil {
				logger.Error("更新订单状态失败", zap.Int64("orderId", o.ID), zap.Error(err))
			}
			s.invalidateOrderDetailCache(ctx, o.OrderNo)
			successCount++
		} else {
			if newRetryCount >= maxRetryCount {
				errorMessage = errorMessage + "（已达重试上限，转为最终失败）"
				finalFailCount++
			}
			updates := map[string]interface{}{
				"retry_count":   newRetryCount,
				"error_message": errorMessage,
			}
			if err := s.refundRepo.Update(ctx, rr.ID, updates); err != nil {
				logger.Error("更新退款重试次数失败", zap.Int64("refundId", rr.ID), zap.Error(err))
			}
		}
	}

	logger.Debug("退款失败重试完成",
		zap.Int("total", len(list)),
		zap.Int("success", successCount),
		zap.Int("finalFail", finalFailCount))
	return nil
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
		ReasonType:      r.ReasonType,
		Reason:          r.Reason,
		UsedDays:        r.UsedDays,
		UsedCredits:     r.UsedCredits,
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
	return s.userRepo.FindUsernameByID(ctx, userID)
}

func toMyOrderVO(o *model.SysOrder) vo.MyOrderVO {
	v := vo.MyOrderVO{
		ID:            o.ID,
		OrderNo:       o.OrderNo,
		PackageName:   o.PackageName,
		PackageType:   o.PackageType,
		PackageLevel:  o.PackageLevel,
		CreditAmount:  o.CreditAmount,
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
	return fmt.Sprintf("DH%s%06d", time.Now().Format("20060102150405"), rand.Intn(1000000))
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

// orderPeriodDays 订单有效期天数（积分卡为 NULL 记 0）
func orderPeriodDays(o *model.SysOrder) int {
	if o.PeriodDays == nil {
		return 0
	}
	return *o.PeriodDays
}

func creditAmountPtr(v sql.NullInt64) *int64 {
	if !v.Valid {
		return nil
	}
	amount := v.Int64
	return &amount
}

func periodDaysPtr(v sql.NullInt64) *int {
	if !v.Valid {
		return nil
	}
	days := int(v.Int64)
	return &days
}

// memberLevelRank 等级序（用于会员卡期间只升不降判断）
func memberLevelRank(levelCode string) int {
	rank, ok := map[string]int{"level_0": 0, "level_1": 1, "level_2": 2, "level_3": 3}[levelCode]
	if !ok {
		return 0
	}
	return rank
}

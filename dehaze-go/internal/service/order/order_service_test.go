package order

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	memberrepo "github.com/earthyzinc/dehaze-go/internal/repository/member"
	orderrepo "github.com/earthyzinc/dehaze-go/internal/repository/order"
	pkgsalerepo "github.com/earthyzinc/dehaze-go/internal/repository/pkgsale"
	userrepo "github.com/earthyzinc/dehaze-go/internal/repository/user"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/database"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// newOrderService 组装 OrderService：
// repository 全部注入真实实现（NewTestDB 单事务 + 回滚），
// 外部依赖（支付渠道网关 paymentSvc / 缓存 / 审计 / 会员缓存失效）按构造注入置 nil，
// 仅在用例明确涉及的路径被调用时（如 Pay 非 balance 分支）才需要 mock。
func newOrderService(t *testing.T, db *gorm.DB) *OrderService {
	t.Helper()
	return NewOrderService(
		db,
		orderrepo.NewOrderRepository(db),
		orderrepo.NewPaymentRecordRepository(db),
		orderrepo.NewRefundRecordRepository(db),
		orderrepo.NewAutoRenewRepository(db),
		pkgsalerepo.NewPackageRepository(db),
		pkgsalerepo.NewCouponRepository(db),
		pkgsalerepo.NewUserCouponRepository(db),
		userrepo.NewUserRepository(db),
		memberrepo.NewMemberRepository(db),
		memberrepo.NewMemberBenefitRepository(db),
		nil, // paymentSvc：外部支付网关，支付完成事务不经过渠道，置 nil
		nil, // cache：不启用 Redis，跳过回调锁
		nil, // auditLogSvc
		nil, // memberCacheInvalidator
	)
}

func uniqueName(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
}

// createTestPackage 自建在售套餐（金额单位：分）
func createTestPackage(t *testing.T, db *gorm.DB, levelCode string, salePrice int64) *model.SysPackage {
	t.Helper()
	pkg := &model.SysPackage{
		Name:          uniqueName("test-pkg"),
		LevelCode:     levelCode,
		Period:        "month",
		PeriodDays:    30,
		OriginalPrice: salePrice * 2,
		SalePrice:     salePrice,
		Description:   "测试套餐",
		Status:        1,
	}
	require.NoError(t, db.Create(pkg).Error)
	return pkg
}

// createTestCoupon 自建无门槛优惠券模板
func createTestCoupon(t *testing.T, db *gorm.DB, faceValue int64) *model.SysCoupon {
	t.Helper()
	c := &model.SysCoupon{
		Name:      uniqueName("test-coupon"),
		Type:      "no_threshold",
		FaceValue: faceValue,
		ValidType: "fixed",
		Status:    1,
	}
	require.NoError(t, db.Create(c).Error)
	return c
}

// createTestUserCoupon 自建未使用状态的用户优惠券实例
func createTestUserCoupon(t *testing.T, db *gorm.DB, userID, couponID int64) *model.SysUserCoupon {
	t.Helper()
	uc := &model.SysUserCoupon{
		UserID:      userID,
		CouponID:    couponID,
		Status:      1,
		ReceiveTime: time.Now(),
	}
	require.NoError(t, db.Create(uc).Error)
	return uc
}

// createTestMember 自建会员记录
func createTestMember(t *testing.T, db *gorm.DB, userID int64, levelCode string, expire *time.Time) *model.SysMember {
	t.Helper()
	m := &model.SysMember{
		UserID:              userID,
		LevelCode:           levelCode,
		LevelSource:         "growth",
		TotalConsumption:    0,
		ExpireTime:          expire,
		MonthlyDehazeQuota:  20,
		MonthlyDehazeUsed:   0,
		MonthlyEvaluateQuota: 20,
		MonthlyEvaluateUsed:  0,
		Status:              1,
	}
	require.NoError(t, db.Create(m).Error)
	return m
}

// createTestOrder 自建待支付订单
func createTestOrder(t *testing.T, db *gorm.DB, userID, pkgID int64, couponID *int64, couponAmount, payableAmount int64) *model.SysOrder {
	t.Helper()
	o := &model.SysOrder{
		OrderNo:        uniqueName("TEST"),
		UserID:         userID,
		PackageID:      pkgID,
		PackageName:    "test-pkg",
		PackageLevel:   "level_1",
		PeriodDays:     30,
		OriginalPrice:  19900,
		DiscountAmount: 0,
		CouponID:       couponID,
		CouponAmount:   couponAmount,
		PayableAmount:  payableAmount,
		PaidAmount:     0,
		Status:         1,
		ExpireTime:     time.Now().Add(30 * time.Minute),
		IsAutoRenew:    0,
	}
	require.NoError(t, db.Create(o).Error)
	return o
}

func getOrder(t *testing.T, db *gorm.DB, orderNo string) model.SysOrder {
	t.Helper()
	var o model.SysOrder
	require.NoError(t, db.Where("order_no = ?", orderNo).First(&o).Error)
	return o
}

// TestHandlePaymentCallback_Success_FullChain 正常完成：
// 支付回调后订单状态、优惠券核销、会员等级/余额/配额更新在真实 DB 中一致落库。
func TestHandlePaymentCallback_Success_FullChain(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc := newOrderService(t, db)

	userID := int64(900001)
	memberExpire := time.Now().Add(10 * 24 * time.Hour).Truncate(time.Second)
	pkg := createTestPackage(t, db, "level_1", 9900) // 9900 分 = 99 元
	coupon := createTestCoupon(t, db, 1000)          // 10 元无门槛券
	uc := createTestUserCoupon(t, db, userID, coupon.ID)
	_ = createTestMember(t, db, userID, "level_0", &memberExpire)
	order := createTestOrder(t, db, userID, pkg.ID, &uc.ID, 1000, 8900)

	ctx := context.Background()
	err := svc.HandlePaymentCallback(ctx, "wechat", order.OrderNo, "wx_cb_001", 8900, true, "raw-callback")
	require.NoError(t, err)

	// 订单：状态流转 + 实付金额 + 权益时间顺延
	saved := getOrder(t, db, order.OrderNo)
	assert.Equal(t, int8(2), saved.Status, "订单应置为已支付")
	assert.Equal(t, int64(8900), saved.PaidAmount)
	require.NotNil(t, saved.PayMethod)
	assert.Equal(t, "wechat", *saved.PayMethod)
	require.NotNil(t, saved.PaidTime)
	require.NotNil(t, saved.EffectiveTime)
	require.NotNil(t, saved.PackageExpireTime)
	// 会员未到期 → 权益从原到期日顺延
	assert.WithinDuration(t, memberExpire, *saved.EffectiveTime, time.Second)
	assert.WithinDuration(t, memberExpire.AddDate(0, 0, 30), *saved.PackageExpireTime, time.Second)

	// 优惠券：核销
	var savedUC model.SysUserCoupon
	require.NoError(t, db.First(&savedUC, uc.ID).Error)
	assert.Equal(t, int8(2), savedUC.Status, "用户优惠券应核销为已使用")
	require.NotNil(t, savedUC.UsedTime)
	require.NotNil(t, savedUC.UsedOrderID)
	assert.Equal(t, saved.ID, *savedUC.UsedOrderID)

	// 优惠券模板：used_qty 递增
	var savedCoupon model.SysCoupon
	require.NoError(t, db.First(&savedCoupon, coupon.ID).Error)
	assert.Equal(t, 1, savedCoupon.UsedQty)

	// 会员：等级/来源/累计消费/配额更新
	var savedMember model.SysMember
	require.NoError(t, db.Where("user_id = ?", userID).First(&savedMember).Error)
	assert.Equal(t, "level_1", savedMember.LevelCode)
	assert.Equal(t, "package", savedMember.LevelSource)
	assert.Equal(t, int64(8900), savedMember.TotalConsumption, "累计消费应加上实付金额")
	assert.Equal(t, 100, savedMember.MonthlyDehazeQuota, "level_1 权益去雾配额")
	assert.Equal(t, 100, savedMember.MonthlyEvaluateQuota)
	assert.Equal(t, int8(1), savedMember.Status)
	require.NotNil(t, savedMember.ExpireTime)
	assert.WithinDuration(t, memberExpire.AddDate(0, 0, 30), *savedMember.ExpireTime, time.Second)

	// 支付流水
	var payments []model.SysPaymentRecord
	require.NoError(t, db.Where("order_id = ?", saved.ID).Find(&payments).Error)
	require.Len(t, payments, 1)
	assert.Equal(t, "wechat", payments[0].Channel)
	assert.Equal(t, int64(8900), payments[0].Amount)
	assert.Equal(t, int8(2), payments[0].Status)
	assert.Equal(t, "wx_cb_001", payments[0].PaymentNo)
}

// TestHandlePaymentCallback_Rollback_WhenMemberMissing 部分失败回滚（核心用例）：
// 在真实 DB 内制造约束冲突（订单存在但 sys_member 不存在，updateMemberAfterPaymentInTx
// 返回 MEMBER_NOT_FOUND），事务中段失败后断言订单/优惠券/会员/支付流水全部回滚到初态。
func TestHandlePaymentCallback_Rollback_WhenMemberMissing(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc := newOrderService(t, db)

	userID := int64(900002)
	pkg := createTestPackage(t, db, "level_1", 9900)
	coupon := createTestCoupon(t, db, 1000)
	uc := createTestUserCoupon(t, db, userID, coupon.ID)
	order := createTestOrder(t, db, userID, pkg.ID, &uc.ID, 1000, 8900)
	// 故意不创建 sys_member 记录

	ctx := context.Background()
	err := svc.HandlePaymentCallback(ctx, "wechat", order.OrderNo, "wx_cb_002", 8900, true, "raw")
	require.Error(t, err)
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok)
	assert.Equal(t, common.DATABASE_ERROR, bizErr.Code(), "中段失败应被包装为数据库错误")

	// 订单：回滚到待支付初态
	saved := getOrder(t, db, order.OrderNo)
	assert.Equal(t, int8(1), saved.Status)
	assert.Equal(t, int64(0), saved.PaidAmount)
	assert.Nil(t, saved.PaidTime)
	assert.Nil(t, saved.PayMethod)
	assert.Nil(t, saved.EffectiveTime)

	// 优惠券：未核销
	var savedUC model.SysUserCoupon
	require.NoError(t, db.First(&savedUC, uc.ID).Error)
	assert.Equal(t, int8(1), savedUC.Status)
	assert.Nil(t, savedUC.UsedOrderID)
	assert.Nil(t, savedUC.UsedTime)

	// 优惠券模板 used_qty 未增加
	var savedCoupon model.SysCoupon
	require.NoError(t, db.First(&savedCoupon, coupon.ID).Error)
	assert.Equal(t, 0, savedCoupon.UsedQty)

	// 会员：仍无记录（无副作用）
	var memberCount int64
	require.NoError(t, db.Model(&model.SysMember{}).Where("user_id = ?", userID).Count(&memberCount).Error)
	assert.Equal(t, int64(0), memberCount)

	// 支付流水与订单/优惠券/会员同事务创建，中段失败应一并回滚
	var payments []model.SysPaymentRecord
	require.NoError(t, db.Where("order_id = ?", saved.ID).Find(&payments).Error)
	assert.Len(t, payments, 0, "支付流水应在事务内创建，中段失败随事务回滚")
}

// TestHandlePaymentCallback_OneCentDecimal DECIMAL 精度边界：0.01 元（1 分）订单，
// 全程金额一致落库且累计消费精确 +1 分。
func TestHandlePaymentCallback_OneCentDecimal(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc := newOrderService(t, db)

	userID := int64(900003)
	pkg := createTestPackage(t, db, "level_1", 1) // 0.01 元
	_ = createTestMember(t, db, userID, "level_0", nil)
	order := createTestOrder(t, db, userID, pkg.ID, nil, 0, 1)

	ctx := context.Background()
	err := svc.HandlePaymentCallback(ctx, "balance", order.OrderNo, "bal_cb_003", 1, true, "")
	require.NoError(t, err)

	saved := getOrder(t, db, order.OrderNo)
	assert.Equal(t, int64(1), saved.PaidAmount, "实付金额应为 1 分")

	var payments []model.SysPaymentRecord
	require.NoError(t, db.Where("order_id = ?", saved.ID).Find(&payments).Error)
	require.Len(t, payments, 1)
	assert.Equal(t, int64(1), payments[0].Amount)

	var savedMember model.SysMember
	require.NoError(t, db.Where("user_id = ?", userID).First(&savedMember).Error)
	assert.Equal(t, int64(1), savedMember.TotalConsumption, "累计消费精确 +1 分")
}

// TestHandlePaymentCallback_AmountMismatch 回调金额与订单应付不一致时拒绝，
// 订单状态不变且无支付流水落库。
func TestHandlePaymentCallback_AmountMismatch(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc := newOrderService(t, db)

	userID := int64(900004)
	pkg := createTestPackage(t, db, "level_1", 9900)
	order := createTestOrder(t, db, userID, pkg.ID, nil, 0, 8900)

	ctx := context.Background()
	err := svc.HandlePaymentCallback(ctx, "wechat", order.OrderNo, "wx_cb_004", 9900, true, "raw")
	require.Error(t, err)
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok)
	assert.Equal(t, common.PAYMENT_AMOUNT_MISMATCH, bizErr.Code())

	saved := getOrder(t, db, order.OrderNo)
	assert.Equal(t, int8(1), saved.Status, "金额不一致不应改变订单状态")

	var payments []model.SysPaymentRecord
	require.NoError(t, db.Where("order_id = ?", saved.ID).Find(&payments).Error)
	assert.Len(t, payments, 0, "金额校验在创建流水之前，不应有流水落库")
}

// TestPay_Balance_Success 余额支付入口闭环：支付完成 + 套餐销量递增。
func TestPay_Balance_Success(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc := newOrderService(t, db)

	userID := int64(900005)
	pkg := createTestPackage(t, db, "level_1", 9900)
	_ = createTestMember(t, db, userID, "level_0", nil)
	order := createTestOrder(t, db, userID, pkg.ID, nil, 0, 9900)

	ctx := database.SetUserID(context.Background(), userID)
	result, err := svc.Pay(ctx, order.OrderNo, &bo.PayRequest{PayMethod: "balance"})
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.True(t, result.Paid, "余额支付应直接完成")

	saved := getOrder(t, db, order.OrderNo)
	assert.Equal(t, int8(2), saved.Status)

	var savedPkg model.SysPackage
	require.NoError(t, db.First(&savedPkg, pkg.ID).Error)
	assert.Equal(t, int64(1), savedPkg.SalesCount, "支付成功套餐销量应 +1")
}

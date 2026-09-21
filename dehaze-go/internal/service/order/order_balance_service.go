package order

import (
	"context"
	"fmt"
	"math/rand"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	orderrepo "github.com/earthyzinc/dehaze-go/internal/repository/order"
	paymentsvc "github.com/earthyzinc/dehaze-go/internal/service/payment"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/database"
)

// balanceCASRetry 乐观锁 CAS 失败重试上限（python balance_account_service._CAS_RETRY 同口径）
const balanceCASRetry = 3

// GetBalance 查询用户余额账户（不存在则初始化零账户）
func (s *OrderService) GetBalance(ctx context.Context, userID int64) (*vo.BalanceVO, error) {
	account, err := orderrepo.NewBalanceAccountRepository(s.db).GetOrCreate(ctx, userID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询余额账户失败", err)
	}
	return &vo.BalanceVO{Balance: account.Balance, FrozenBalance: account.FrozenBalance}, nil
}

// applyBalanceRefund 提交余额退款申请（充值余额退回）。
// 申请校验可用余额与待审核申请唯一性，余额/冻结终验留待管理员审核环节。
func (s *OrderService) ApplyBalanceRefund(ctx context.Context, userID int64, form *bo.BalanceRefundForm) (*vo.BalanceRefundResult, error) {
	account, err := orderrepo.NewBalanceAccountRepository(s.db).GetOrCreate(ctx, userID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询余额账户失败", err)
	}

	amount := account.Balance
	if form.Amount != nil && *form.Amount > 0 {
		amount = *form.Amount
	}
	if amount > account.Balance {
		return nil, common.NewBizError(common.BALANCE_INSUFFICIENT, "余额不足")
	}

	pending, err := orderrepo.NewBalanceRefundRepository(s.db).FindPendingByUserID(ctx, userID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询余额退款申请失败", err)
	}
	if pending != nil {
		return nil, common.NewBizError(common.OPERATION_NOT_ALLOW, "已存在待审核的余额退款申请")
	}

	record := &model.SysBalanceRefund{
		RefundNo:  fmt.Sprintf("BR%s%03d", time.Now().Format("20060102150405"), rand.Intn(1000)),
		UserID:    userID,
		Amount:    amount,
		Status:    1,
		ApplyTime: time.Now(),
	}
	if err := orderrepo.NewBalanceRefundRepository(s.db).Create(ctx, record); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建余额退款申请失败", err)
	}

	if s.auditLogSvc != nil {
		s.auditLogSvc.RecordAuditAsync(ctx, userID, "balance_refund", record.RefundNo, "apply", "order", nil, form, database.GetIP(ctx), database.GetUserAgent(ctx))
	}
	return &vo.BalanceRefundResult{RefundNo: record.RefundNo, Amount: record.Amount}, nil
}

// AuditBalanceRefund 管理员审核余额退款：校验余额与冻结，原路退回渠道后扣减可用余额。
func (s *OrderService) AuditBalanceRefund(ctx context.Context, auditorID, refundID int64, form *bo.BalanceRefundAuditForm) error {
	refundRepo := orderrepo.NewBalanceRefundRepository(s.db)
	record, err := refundRepo.FindByID(ctx, refundID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询余额退款记录失败", err)
	}
	if record == nil {
		return common.NewBizError(common.REFUND_NOT_FOUND, "余额退款记录不存在")
	}
	if record.Status != 1 {
		return common.NewBizError(common.ORDER_STATUS_INVALID, "退款状态不允许此操作")
	}

	account, err := orderrepo.NewBalanceAccountRepository(s.db).GetOrCreate(ctx, record.UserID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询余额账户失败", err)
	}
	if account.Balance < record.Amount {
		return common.NewBizError(common.BALANCE_INSUFFICIENT, "余额不足")
	}
	if account.FrozenBalance > 0 {
		return common.NewBizError(common.ORDER_STATUS_INVALID, "存在冻结余额，暂不可退")
	}

	now := time.Now()
	updates := map[string]interface{}{
		"audit_time":   now,
		"auditor_id":   auditorID,
		"audit_remark": form.Remark,
	}

	// 原路退回（渠道未指定/未启用时直接成功，与 python 渠道 mock 口径一致）
	var channelRefundNo string
	refundErr := error(nil)
	if form.Channel == "wechat" || form.Channel == "alipay" {
		if s.paymentSvc != nil {
			result, err := s.paymentSvc.Refund(ctx, &paymentsvc.RefundRequest{
				OrderNo:   record.RefundNo,
				PaymentNo: record.RefundNo,
				Channel:   form.Channel,
				Amount:    record.Amount,
				Reason:    "余额退款",
			})
			if err != nil {
				refundErr = err
			} else if !result.Success {
				msg := result.ErrorMessage
				if msg == "" {
					msg = "渠道退款失败"
				}
				refundErr = common.NewBizError(common.CALL_THIRD_PARTY_SERVICE_ERROR, msg)
			} else {
				channelRefundNo = result.RefundNo
			}
		}
	}

	if refundErr == nil {
		if err := s.withdrawBalance(ctx, record.UserID, record.Amount, record.ID); err != nil {
			refundErr = err
		}
	}

	if refundErr != nil {
		updates["status"] = 3
		updates["error_message"] = refundErr.Error()
		if err := refundRepo.Update(ctx, refundID, updates); err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "更新余额退款记录失败", err)
		}
		return refundErr
	}

	updates["status"] = 2
	updates["refund_time"] = now
	updates["channel"] = form.Channel
	updates["channel_refund_no"] = channelRefundNo
	if err := refundRepo.Update(ctx, refundID, updates); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新余额退款记录失败", err)
	}

	if s.auditLogSvc != nil {
		s.auditLogSvc.RecordAuditAsync(ctx, auditorID, "balance_refund", refundID, "balance_refund_approve", "order", nil, form, database.GetIP(ctx), database.GetUserAgent(ctx))
	}
	return nil
}

// withdrawBalance 原路退回后扣减可用余额（CAS 重试，与 python withdraw 口径一致）
func (s *OrderService) withdrawBalance(ctx context.Context, userID, amount int64, relatedID int64) error {
	accountRepo := orderrepo.NewBalanceAccountRepository(s.db)
	logRepo := orderrepo.NewBalanceLogRepository(s.db)
	for i := 0; i < balanceCASRetry; i++ {
		account, err := accountRepo.GetOrCreate(ctx, userID)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询余额账户失败", err)
		}
		if account.Balance < amount {
			return common.NewBizError(common.BALANCE_INSUFFICIENT, "余额不足")
		}
		ok, err := accountRepo.AdjustBalance(ctx, userID, -amount, account.Version)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "扣减余额失败", err)
		}
		if ok {
			if err := logRepo.Create(ctx, &model.SysBalanceLog{
				UserID:       userID,
				ChangeType:   "refund",
				Amount:       -amount,
				BalanceAfter: account.Balance - amount,
				RelatedID:    &relatedID,
			}); err != nil {
				return common.WrapBizError(common.DATABASE_ERROR, "写入余额流水失败", err)
			}
			return nil
		}
	}
	return common.NewBizError(common.BUSINESS_ERROR, "余额退款扣减失败，请重试")
}

// CreateRecharge 创建余额充值订单（渠道统一下单，回调入账）
func (s *OrderService) CreateRecharge(ctx context.Context, userID int64, form *bo.RechargeCreateForm) (*vo.RechargeResult, error) {
	rechargeNo := fmt.Sprintf("RC%s%06d", time.Now().Format("20060102150405"), rand.Intn(1000000))

	var payURL, qrCode *string
	if s.paymentSvc != nil {
		payResult, err := s.paymentSvc.CreateOrder(ctx, &paymentsvc.UnifiedOrderRequest{
			OrderNo:     rechargeNo,
			Amount:      form.Amount,
			Description: "余额充值",
			PayMethod:   form.PayMethod,
		})
		if err != nil {
			return nil, common.WrapBizError(common.CALL_THIRD_PARTY_SERVICE_ERROR, "渠道下单失败", err)
		}
		payURL = &payResult.PayURL
		qrCode = &payResult.QRCode
	}

	record := &model.SysRecharge{
		RechargeNo: rechargeNo,
		UserID:     userID,
		Amount:     form.Amount,
		PayMethod:  form.PayMethod,
		Status:     1,
	}
	if err := orderrepo.NewRechargeRepository(s.db).Create(ctx, record); err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "创建充值订单失败", err)
	}

	if s.auditLogSvc != nil {
		s.auditLogSvc.RecordAuditAsync(ctx, userID, "recharge", rechargeNo, "create", "order", nil, form, database.GetIP(ctx), database.GetUserAgent(ctx))
	}
	return &vo.RechargeResult{RechargeNo: rechargeNo, PayMethod: form.PayMethod, PayURL: payURL, QRCode: qrCode}, nil
}

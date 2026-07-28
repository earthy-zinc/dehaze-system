package order

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

type IOrderService interface {
	Create(ctx context.Context, userID int64, form *bo.OrderCreateForm) (*vo.PayResult, error)
	ListMy(ctx context.Context, userID int64, q *query.MyOrderQuery) (*vo.PageResult[vo.MyOrderVO], error)
	GetDetail(ctx context.Context, orderNo string) (*vo.OrderDetailVO, error)
	Cancel(ctx context.Context, orderNo string, reason string) error
	Pay(ctx context.Context, orderNo string, req *bo.PayRequest) (*vo.PayResult, error)
	ApplyRefund(ctx context.Context, userID int64, orderNo string, form *bo.RefundApplyForm) error

	UpdateAutoRenewConfig(ctx context.Context, userID int64, form *bo.AutoRenewConfigForm) error
	GetAutoRenewConfig(ctx context.Context, userID, packageID int64) (*vo.AutoRenewConfigVO, error)

	GetPage(ctx context.Context, q *query.OrderPageQuery) (*vo.PageResult[vo.OrderPageVO], error)
	ListRefunds(ctx context.Context, q *query.RefundPageQuery) (*vo.PageResult[vo.RefundRecordVO], error)
	ApproveRefund(ctx context.Context, auditorID, refundID int64, form *bo.RefundAuditForm) error
	RejectRefund(ctx context.Context, auditorID, refundID int64, form *bo.RefundAuditForm) error
	GetStats(ctx context.Context, startTime, endTime string) (*vo.OrderStatsVO, error)

	HandlePaymentCallback(ctx context.Context, channel, orderNo, channelNo string, amount int64, success bool, rawContent string) error
}

type IOrderJobRunner interface {
	CancelExpiredOrders(ctx context.Context) error
	CompleteExpiredOrders(ctx context.Context) error
	ProcessAutoRenewals(ctx context.Context) error
	ExpireUserCoupons(ctx context.Context) error
}

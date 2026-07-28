package order

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
)

type OrderWithUser struct {
	model.SysOrder
	Username string `gorm:"column:username" json:"username"`
}

type RefundWithOrder struct {
	model.SysRefundRecord
	OrderNo  string `gorm:"column:order_no" json:"orderNo"`
	Username string `gorm:"column:username" json:"username"`
}

type PackageStatRow struct {
	PackageID   int64  `gorm:"column:package_id" json:"packageId"`
	PackageName string `gorm:"column:package_name" json:"packageName"`
	Count       int64  `gorm:"column:count" json:"count"`
	Revenue     int64  `gorm:"column:revenue" json:"revenue"`
}

type DailyStatRow struct {
	Date    string `gorm:"column:date" json:"date"`
	Count   int64  `gorm:"column:count" json:"count"`
	Revenue int64  `gorm:"column:revenue" json:"revenue"`
}

type IOrderRepository interface {
	FindByID(ctx context.Context, id int64) (*model.SysOrder, error)
	FindByOrderNo(ctx context.Context, orderNo string) (*model.SysOrder, error)
	FindByIDs(ctx context.Context, ids []int64) ([]model.SysOrder, error)
	Create(ctx context.Context, o *model.SysOrder) error
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
	UpdateByOrderNo(ctx context.Context, orderNo string, updates map[string]interface{}) error
	FindPageMyOrders(ctx context.Context, userID int64, q *query.MyOrderQuery) ([]OrderWithUser, int64, error)
	FindPage(ctx context.Context, q *query.OrderPageQuery) ([]OrderWithUser, int64, error)
	FindPendingExpired(ctx context.Context, before time.Time) ([]model.SysOrder, error)
	FindPaidExpired(ctx context.Context, before time.Time) ([]model.SysOrder, error)
	CountByStatus(ctx context.Context, status int8) (int64, error)
	SumRevenue(ctx context.Context, startTime, endTime string) (int64, error)
	SumRefund(ctx context.Context, startTime, endTime string) (int64, error)
	CountTotalOrders(ctx context.Context, startTime, endTime string) (int64, error)
	CountByPayMethod(ctx context.Context, payMethod string) (int64, error)
	GetPackageDistribution(ctx context.Context, startTime, endTime string) ([]PackageStatRow, error)
	GetDailyStats(ctx context.Context, startTime, endTime string) ([]DailyStatRow, error)
}

type IPaymentRecordRepository interface {
	Create(ctx context.Context, p *model.SysPaymentRecord) error
	FindByOrderID(ctx context.Context, orderID int64) ([]model.SysPaymentRecord, error)
	UpdateStatus(ctx context.Context, id int64, status int8, callbackContent, errMsg string) error
}

type IRefundRecordRepository interface {
	FindByID(ctx context.Context, id int64) (*model.SysRefundRecord, error)
	FindByOrderID(ctx context.Context, orderID int64) (*model.SysRefundRecord, error)
	Create(ctx context.Context, rr *model.SysRefundRecord) error
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
	FindPage(ctx context.Context, q *query.RefundPageQuery) ([]RefundWithOrder, int64, error)
}

type IAutoRenewRepository interface {
	FindByUserIDAndPackageID(ctx context.Context, userID, packageID int64) (*model.SysAutoRenew, error)
	Create(ctx context.Context, ar *model.SysAutoRenew) error
	Update(ctx context.Context, id int64, updates map[string]interface{}) error
	Upsert(ctx context.Context, ar *model.SysAutoRenew) error
	FindDueRenewals(ctx context.Context, before interface{}) ([]model.SysAutoRenew, error)
}

func orderStatusToInt(status string) int8 {
	switch status {
	case "pending":
		return 1
	case "paid":
		return 2
	case "completed":
		return 3
	case "cancelled":
		return 4
	case "refunding":
		return 5
	case "refunded":
		return 6
	}
	return 0
}

func OrderStatusToString(status int8) string {
	switch status {
	case 1:
		return "pending"
	case 2:
		return "paid"
	case 3:
		return "completed"
	case 4:
		return "cancelled"
	case 5:
		return "refunding"
	case 6:
		return "refunded"
	}
	return ""
}

func RefundStatusToString(status int8) string {
	switch status {
	case 1:
		return "refunding"
	case 2:
		return "refunded"
	case 3:
		return "refund_failed"
	}
	return ""
}

func refundStatusToInt(status string) int8 {
	switch status {
	case "refunding":
		return 1
	case "refunded":
		return 2
	case "refund_failed":
		return 3
	}
	return 0
}

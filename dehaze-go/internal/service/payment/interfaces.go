package payment

import "context"

type UnifiedOrderRequest struct {
	OrderNo     string
	Amount      int64
	Description string
	PayMethod   string
	NotifyURL   string
}

type UnifiedOrderResult struct {
	OrderNo   string
	PayURL    string
	QRCode    string
	PrepayID  string
	Paid      bool
}

type CallbackResult struct {
	OrderNo       string
	ChannelNo     string
	Amount        int64
	Success       bool
	RawContent    string
	ErrorResponse string
}

type RefundRequest struct {
	OrderNo   string
	PaymentNo string
	Channel   string
	Amount    int64
	Reason    string
}

type RefundResult struct {
	RefundNo     string
	Success      bool
	ErrorMessage string
}

type QueryResult struct {
	OrderNo string
	Success bool
	Amount  int64
}

type IPaymentChannelService interface {
	CreateOrder(ctx context.Context, req *UnifiedOrderRequest) (*UnifiedOrderResult, error)
	VerifyCallback(ctx context.Context, channel string, body []byte) (*CallbackResult, error)
	Refund(ctx context.Context, req *RefundRequest) (*RefundResult, error)
	QueryOrder(ctx context.Context, orderNo string) (*QueryResult, error)
	CloseOrder(ctx context.Context, orderNo string) error
}

type IChannelAdapter interface {
	CreateOrder(ctx context.Context, req *UnifiedOrderRequest) (*UnifiedOrderResult, error)
	VerifyCallback(ctx context.Context, body []byte) (*CallbackResult, error)
	Refund(ctx context.Context, req *RefundRequest) (*RefundResult, error)
	QueryOrder(ctx context.Context, orderNo string) (*QueryResult, error)
	CloseOrder(ctx context.Context, orderNo string) error
	Enabled() bool
}

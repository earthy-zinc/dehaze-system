package payment

import (
	"context"
	"fmt"
	"time"
)

type MockChannelAdapter struct{}

func NewMockChannelAdapter() *MockChannelAdapter {
	return &MockChannelAdapter{}
}

func (a *MockChannelAdapter) CreateOrder(ctx context.Context, req *UnifiedOrderRequest) (*UnifiedOrderResult, error) {
	return &UnifiedOrderResult{
		OrderNo:  req.OrderNo,
		PayURL:   fmt.Sprintf("mock://pay/%s?t=%d", req.OrderNo, time.Now().Unix()),
		QRCode:   fmt.Sprintf("mock://qr/%s", req.OrderNo),
		PrepayID: fmt.Sprintf("mock_prepay_%s", req.OrderNo),
	}, nil
}

func (a *MockChannelAdapter) VerifyCallback(ctx context.Context, body []byte) (*CallbackResult, error) {
	return &CallbackResult{
		Success:    true,
		RawContent: string(body),
	}, nil
}

func (a *MockChannelAdapter) Refund(ctx context.Context, req *RefundRequest) (*RefundResult, error) {
	return &RefundResult{
		RefundNo: fmt.Sprintf("mock_refund_%s_%d", req.OrderNo, time.Now().Unix()),
		Success:  true,
	}, nil
}

func (a *MockChannelAdapter) QueryOrder(ctx context.Context, orderNo string) (*QueryResult, error) {
	return &QueryResult{OrderNo: orderNo, Success: false}, nil
}

func (a *MockChannelAdapter) CloseOrder(ctx context.Context, orderNo string) error {
	return nil
}

func (a *MockChannelAdapter) Enabled() bool {
	return false
}

var _ IChannelAdapter = (*MockChannelAdapter)(nil)

package payment

import (
	"context"
	"fmt"

	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

type PaymentChannelService struct {
	adapters map[string]IChannelAdapter
}

func NewPaymentChannelService(cfg options.Payment) *PaymentChannelService {
	svc := &PaymentChannelService{
		adapters: make(map[string]IChannelAdapter),
	}

	mock := NewMockChannelAdapter()

	wechat := NewWechatPayAdapter(cfg.Wechat)
	if wechat.Enabled() {
		svc.adapters["wechat"] = wechat
		logger.Info("支付渠道已启用: wechat")
	} else {
		svc.adapters["wechat"] = mock
	}

	alipay := NewAlipayAdapter(cfg.Alipay)
	if alipay.Enabled() {
		svc.adapters["alipay"] = alipay
		logger.Info("支付渠道已启用: alipay")
	} else {
		svc.adapters["alipay"] = mock
	}

	svc.adapters["mock"] = mock
	return svc
}

func (s *PaymentChannelService) getAdapter(channel string) (IChannelAdapter, error) {
	adapter, ok := s.adapters[channel]
	if !ok || adapter == nil {
		return nil, fmt.Errorf("不支持的支付渠道: %s", channel)
	}
	return adapter, nil
}

func (s *PaymentChannelService) CreateOrder(ctx context.Context, req *UnifiedOrderRequest) (*UnifiedOrderResult, error) {
	adapter, err := s.getAdapter(req.PayMethod)
	if err != nil {
		return nil, err
	}
	result, err := adapter.CreateOrder(ctx, req)
	if err != nil {
		logger.Error("支付渠道下单失败", zap.String("channel", req.PayMethod), zap.String("orderNo", req.OrderNo), zap.Error(err))
		return nil, fmt.Errorf("下单失败: %w", err)
	}
	return result, nil
}

func (s *PaymentChannelService) VerifyCallback(ctx context.Context, channel string, body []byte) (*CallbackResult, error) {
	adapter, err := s.getAdapter(channel)
	if err != nil {
		return nil, err
	}
	result, err := adapter.VerifyCallback(ctx, body)
	if err != nil {
		logger.Error("支付回调验签失败", zap.String("channel", channel), zap.Error(err))
		return nil, fmt.Errorf("回调验签失败: %w", err)
	}
	return result, nil
}

func (s *PaymentChannelService) Refund(ctx context.Context, req *RefundRequest) (*RefundResult, error) {
	channel := req.Channel
	if channel == "" {
		channel = "mock"
	}
	adapter, err := s.getAdapter(channel)
	if err != nil {
		return nil, err
	}
	result, err := adapter.Refund(ctx, req)
	if err != nil {
		logger.Error("支付渠道退款失败", zap.String("channel", channel), zap.String("orderNo", req.OrderNo), zap.Error(err))
		return nil, fmt.Errorf("退款失败: %w", err)
	}
	return result, nil
}

func (s *PaymentChannelService) QueryOrder(ctx context.Context, orderNo string) (*QueryResult, error) {
	return nil, fmt.Errorf("QueryOrder 需通过 ChannelQueryOrder 指定渠道")
}

func (s *PaymentChannelService) CloseOrder(ctx context.Context, orderNo string) error {
	return nil
}

func (s *PaymentChannelService) ChannelQueryOrder(ctx context.Context, channel, orderNo string) (*QueryResult, error) {
	adapter, err := s.getAdapter(channel)
	if err != nil {
		return nil, err
	}
	return adapter.QueryOrder(ctx, orderNo)
}

func (s *PaymentChannelService) ChannelCloseOrder(ctx context.Context, channel, orderNo string) error {
	adapter, err := s.getAdapter(channel)
	if err != nil {
		return err
	}
	return adapter.CloseOrder(ctx, orderNo)
}

func (s *PaymentChannelService) ChannelEnabled(channel string) bool {
	adapter, ok := s.adapters[channel]
	if !ok {
		return false
	}
	return adapter.Enabled()
}

var _ IPaymentChannelService = (*PaymentChannelService)(nil)

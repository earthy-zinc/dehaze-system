package payment

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

type WechatPayAdapter struct {
	cfg    options.WechatPayConfig
	client *http.Client
}

func NewWechatPayAdapter(cfg options.WechatPayConfig) *WechatPayAdapter {
	return &WechatPayAdapter{
		cfg: cfg,
		client: &http.Client{
			Timeout: 15 * time.Second,
		},
	}
}

func (a *WechatPayAdapter) Enabled() bool {
	return a.cfg.Enabled
}

func (a *WechatPayAdapter) CreateOrder(ctx context.Context, req *UnifiedOrderRequest) (*UnifiedOrderResult, error) {
	url := a.cfg.BaseURL + "/v3/pay/transactions/native"
	body := map[string]interface{}{
		"appid":        a.cfg.AppID,
		"mchid":        a.cfg.MchID,
		"description":  req.Description,
		"out_trade_no": req.OrderNo,
		"notify_url":   a.cfg.NotifyURL,
		"amount": map[string]interface{}{
			"total":    req.Amount,
			"currency": "CNY",
		},
	}
	respBody, err := a.doRequest(ctx, "POST", url, body)
	if err != nil {
		return nil, fmt.Errorf("微信下单失败: %w", err)
	}
	var resp struct {
		CodeURL string `json:"code_url"`
	}
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return nil, fmt.Errorf("解析微信下单响应失败: %w", err)
	}
	return &UnifiedOrderResult{
		OrderNo:  req.OrderNo,
		QRCode:   resp.CodeURL,
		PayURL:   resp.CodeURL,
		PrepayID: req.OrderNo,
	}, nil
}

func (a *WechatPayAdapter) VerifyCallback(ctx context.Context, body []byte) (*CallbackResult, error) {
	var cb struct {
		OutTradeNo  string `json:"out_trade_no"`
		TransactionID string `json:"transaction_id"`
		TradeState   string `json:"trade_state"`
		Amount struct {
			Total int64 `json:"total"`
		} `json:"amount"`
	}
	if err := json.Unmarshal(body, &cb); err != nil {
		return nil, fmt.Errorf("解析微信回调失败: %w", err)
	}
	success := cb.TradeState == "SUCCESS"
	return &CallbackResult{
		OrderNo:    cb.OutTradeNo,
		ChannelNo:  cb.TransactionID,
		Amount:     cb.Amount.Total,
		Success:    success,
		RawContent: string(body),
	}, nil
}

func (a *WechatPayAdapter) Refund(ctx context.Context, req *RefundRequest) (*RefundResult, error) {
	url := a.cfg.BaseURL + "/v3/refund/domestic/refunds"
	body := map[string]interface{}{
		"out_trade_no":  req.OrderNo,
		"out_refund_no": fmt.Sprintf("RF%s%d", req.OrderNo, time.Now().Unix()),
		"reason":        req.Reason,
		"amount": map[string]interface{}{
			"refund":   req.Amount,
			"total":    req.Amount,
			"currency": "CNY",
		},
	}
	respBody, err := a.doRequest(ctx, "POST", url, body)
	if err != nil {
		return &RefundResult{Success: false, ErrorMessage: err.Error()}, err
	}
	var resp struct {
		RefundID string `json:"refund_id"`
		Status   string `json:"status"`
	}
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return &RefundResult{Success: false, ErrorMessage: err.Error()}, err
	}
	return &RefundResult{
		RefundNo: resp.RefundID,
		Success:  resp.Status == "SUCCESS" || resp.Status == "PROCESSING",
	}, nil
}

func (a *WechatPayAdapter) QueryOrder(ctx context.Context, orderNo string) (*QueryResult, error) {
	url := fmt.Sprintf("%s/v3/pay/transactions/out-trade-no/%s?mchid=%s", a.cfg.BaseURL, orderNo, a.cfg.MchID)
	respBody, err := a.doRequest(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	var resp struct {
		TradeState string `json:"trade_state"`
		Amount     struct {
			Total int64 `json:"total"`
		} `json:"amount"`
	}
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return nil, err
	}
	return &QueryResult{
		OrderNo: orderNo,
		Success: resp.TradeState == "SUCCESS",
		Amount:  resp.Amount.Total,
	}, nil
}

func (a *WechatPayAdapter) CloseOrder(ctx context.Context, orderNo string) error {
	url := fmt.Sprintf("%s/v3/pay/transactions/out-trade-no/%s/close", a.cfg.BaseURL, orderNo)
	body := map[string]interface{}{"mchid": a.cfg.MchID}
	_, err := a.doRequest(ctx, "POST", url, body)
	if err != nil {
		logger.Warn("微信关单失败", zap.String("orderNo", orderNo), zap.Error(err))
	}
	return nil
}

func (a *WechatPayAdapter) doRequest(ctx context.Context, method, url string, body interface{}) ([]byte, error) {
	var reqBody io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reqBody = bytes.NewReader(data)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := a.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		return respBody, fmt.Errorf("微信支付接口返回错误: status=%d body=%s", resp.StatusCode, string(respBody))
	}
	return respBody, nil
}

var _ IChannelAdapter = (*WechatPayAdapter)(nil)

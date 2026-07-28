package payment

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

type AlipayAdapter struct {
	cfg    options.AlipayConfig
	client *http.Client
}

func NewAlipayAdapter(cfg options.AlipayConfig) *AlipayAdapter {
	return &AlipayAdapter{
		cfg: cfg,
		client: &http.Client{
			Timeout: 15 * time.Second,
		},
	}
}

func (a *AlipayAdapter) Enabled() bool {
	return a.cfg.Enabled
}

func (a *AlipayAdapter) CreateOrder(ctx context.Context, req *UnifiedOrderRequest) (*UnifiedOrderResult, error) {
	params := url.Values{}
	params.Set("app_id", a.cfg.AppID)
	params.Set("method", "alipay.trade.precreate")
	params.Set("charset", "utf-8")
	params.Set("sign_type", "RSA2")
	params.Set("timestamp", time.Now().Format("2006-01-02 15:04:05"))
	params.Set("version", "1.0")
	params.Set("notify_url", a.cfg.NotifyURL)
	bizContent := fmt.Sprintf(`{"out_trade_no":"%s","total_amount":"%s","subject":"%s"}`, req.OrderNo, formatAlipayAmount(req.Amount), req.Description)
	params.Set("biz_content", bizContent)

	respBody, err := a.doRequest(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("支付宝下单失败: %w", err)
	}
	var resp struct {
		AlipayTradePrecreateResponse struct {
			OutTradeNo string `json:"out_trade_no"`
			QRCode     string `json:"qr_code"`
		} `json:"alipay_trade_precreate_response"`
	}
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return nil, fmt.Errorf("解析支付宝下单响应失败: %w", err)
	}
	return &UnifiedOrderResult{
		OrderNo:  resp.AlipayTradePrecreateResponse.OutTradeNo,
		QRCode:   resp.AlipayTradePrecreateResponse.QRCode,
		PayURL:   resp.AlipayTradePrecreateResponse.QRCode,
		PrepayID: req.OrderNo,
	}, nil
}

func (a *AlipayAdapter) VerifyCallback(ctx context.Context, body []byte) (*CallbackResult, error) {
	values, err := url.ParseQuery(string(body))
	if err != nil {
		return nil, fmt.Errorf("解析支付宝回调失败: %w", err)
	}
	tradeStatus := values.Get("trade_status")
	amountStr := values.Get("total_amount")
	amount := parseAlipayAmount(amountStr)
	success := tradeStatus == "TRADE_SUCCESS" || tradeStatus == "TRADE_FINISHED"
	return &CallbackResult{
		OrderNo:    values.Get("out_trade_no"),
		ChannelNo:  values.Get("trade_no"),
		Amount:     amount,
		Success:    success,
		RawContent: string(body),
	}, nil
}

func (a *AlipayAdapter) Refund(ctx context.Context, req *RefundRequest) (*RefundResult, error) {
	params := url.Values{}
	params.Set("app_id", a.cfg.AppID)
	params.Set("method", "alipay.trade.refund")
	params.Set("charset", "utf-8")
	params.Set("sign_type", "RSA2")
	params.Set("timestamp", time.Now().Format("2006-01-02 15:04:05"))
	params.Set("version", "1.0")
	bizContent := fmt.Sprintf(`{"out_trade_no":"%s","refund_amount":"%s","refund_reason":"%s"}`, req.OrderNo, formatAlipayAmount(req.Amount), req.Reason)
	params.Set("biz_content", bizContent)

	respBody, err := a.doRequest(ctx, params)
	if err != nil {
		return &RefundResult{Success: false, ErrorMessage: err.Error()}, err
	}
	var resp struct {
		AlipayTradeRefundResponse struct {
			OutTradeNo  string `json:"out_trade_no"`
			TradeNo     string `json:"trade_no"`
			FundChange  string `json:"fund_change"`
			Code        string `json:"code"`
			Msg         string `json:"msg"`
		} `json:"alipay_trade_refund_response"`
	}
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return &RefundResult{Success: false, ErrorMessage: err.Error()}, err
	}
	success := resp.AlipayTradeRefundResponse.Code == "10000" && resp.AlipayTradeRefundResponse.FundChange == "Y"
	return &RefundResult{
		RefundNo: resp.AlipayTradeRefundResponse.TradeNo,
		Success:  success,
	}, nil
}

func (a *AlipayAdapter) QueryOrder(ctx context.Context, orderNo string) (*QueryResult, error) {
	params := url.Values{}
	params.Set("app_id", a.cfg.AppID)
	params.Set("method", "alipay.trade.query")
	params.Set("charset", "utf-8")
	params.Set("sign_type", "RSA2")
	params.Set("timestamp", time.Now().Format("2006-01-02 15:04:05"))
	params.Set("version", "1.0")
	bizContent := fmt.Sprintf(`{"out_trade_no":"%s"}`, orderNo)
	params.Set("biz_content", bizContent)

	respBody, err := a.doRequest(ctx, params)
	if err != nil {
		return nil, err
	}
	var resp struct {
		AlipayTradeQueryResponse struct {
			OutTradeNo  string `json:"out_trade_no"`
			TradeStatus string `json:"trade_status"`
			TotalAmount string `json:"total_amount"`
		} `json:"alipay_trade_query_response"`
	}
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return nil, err
	}
	return &QueryResult{
		OrderNo: orderNo,
		Success: resp.AlipayTradeQueryResponse.TradeStatus == "TRADE_SUCCESS" || resp.AlipayTradeQueryResponse.TradeStatus == "TRADE_FINISHED",
		Amount:  parseAlipayAmount(resp.AlipayTradeQueryResponse.TotalAmount),
	}, nil
}

func (a *AlipayAdapter) CloseOrder(ctx context.Context, orderNo string) error {
	params := url.Values{}
	params.Set("app_id", a.cfg.AppID)
	params.Set("method", "alipay.trade.close")
	params.Set("charset", "utf-8")
	params.Set("sign_type", "RSA2")
	params.Set("timestamp", time.Now().Format("2006-01-02 15:04:05"))
	params.Set("version", "1.0")
	bizContent := fmt.Sprintf(`{"out_trade_no":"%s"}`, orderNo)
	params.Set("biz_content", bizContent)
	_, err := a.doRequest(ctx, params)
	if err != nil {
		logger.Warn("支付宝关单失败", zap.String("orderNo", orderNo), zap.Error(err))
	}
	return nil
}

func (a *AlipayAdapter) doRequest(ctx context.Context, params url.Values) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", a.cfg.Gateway, bytes.NewReader([]byte(params.Encode())))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

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
		return respBody, fmt.Errorf("支付宝接口返回错误: status=%d body=%s", resp.StatusCode, string(respBody))
	}
	return respBody, nil
}

func formatAlipayAmount(amount int64) string {
	return fmt.Sprintf("%d.%02d", amount/100, amount%100)
}

func parseAlipayAmount(s string) int64 {
	if s == "" {
		return 0
	}
	var yuan int64
	var fen int64
	neg := false
	dotIdx := -1
	for i, c := range s {
		if c == '-' {
			neg = true
			continue
		}
		if c == '.' {
			dotIdx = i
			break
		}
		if c >= '0' && c <= '9' {
			yuan = yuan*10 + int64(c-'0')
		}
	}
	if dotIdx >= 0 {
		fenStr := s[dotIdx+1:]
		if len(fenStr) >= 1 {
			fen = int64(fenStr[0]-'0') * 10
		}
		if len(fenStr) >= 2 {
			fen = int64(fenStr[0]-'0')*10 + int64(fenStr[1]-'0')
		}
	}
	result := yuan*100 + fen
	if neg {
		result = -result
	}
	return result
}

var _ IChannelAdapter = (*AlipayAdapter)(nil)

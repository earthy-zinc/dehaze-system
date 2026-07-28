package api

import (
	"io"
	"net/http"

	orderservice "github.com/earthyzinc/dehaze-go/internal/service/order"
	paymentsvc "github.com/earthyzinc/dehaze-go/internal/service/payment"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type PaymentApi struct {
	paymentSvc paymentsvc.IPaymentChannelService
	orderSvc   orderservice.IOrderService
}

func NewPaymentApi(paymentSvc paymentsvc.IPaymentChannelService, orderSvc orderservice.IOrderService) *PaymentApi {
	return &PaymentApi{paymentSvc: paymentSvc, orderSvc: orderSvc}
}

func (api *PaymentApi) HandleWechatCallback(c *gin.Context) {
	api.handleCallback(c, "wechat")
}

func (api *PaymentApi) HandleAlipayCallback(c *gin.Context) {
	api.handleCallback(c, "alipay")
}

func (api *PaymentApi) handleCallback(c *gin.Context, channel string) {
	ctx := c.Request.Context()

	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		logger.Error("读取支付回调请求体失败", zap.String("channel", channel), zap.Error(err))
		c.JSON(http.StatusBadRequest, gin.H{"code": "FAIL", "message": "读取请求体失败"})
		return
	}

	result, err := api.paymentSvc.VerifyCallback(ctx, channel, body)
	if err != nil {
		logger.Error("支付回调验签失败", zap.String("channel", channel), zap.Error(err))
		c.JSON(http.StatusBadRequest, gin.H{"code": "FAIL", "message": "验签失败"})
		return
	}
	if result == nil {
		c.JSON(http.StatusBadRequest, gin.H{"code": "FAIL", "message": "回调结果为空"})
		return
	}

	if result.ErrorResponse != "" {
		c.JSON(http.StatusOK, gin.H{"code": "FAIL", "message": result.ErrorResponse})
		return
	}

	if err := api.orderSvc.HandlePaymentCallback(ctx, channel, result.OrderNo, result.ChannelNo, result.Amount, result.Success, result.RawContent); err != nil {
		logger.Error("处理支付回调失败",
			zap.String("channel", channel),
			zap.String("orderNo", result.OrderNo),
			zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"code": "FAIL", "message": "处理回调失败"})
		return
	}

	if channel == "alipay" {
		c.String(http.StatusOK, "success")
		return
	}
	c.JSON(http.StatusOK, gin.H{"code": "SUCCESS", "message": "成功"})
}

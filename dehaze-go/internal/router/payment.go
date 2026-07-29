package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

func RegisterPaymentRoutes(rg *gin.RouterGroup, paymentApi *api.PaymentApi) {
	paymentRouter := rg.Group("/orders/payment")
	{
		paymentRouter.POST("/wechat/callback", paymentApi.HandleWechatCallback)
		paymentRouter.POST("/alipay/callback", paymentApi.HandleAlipayCallback)
	}
}

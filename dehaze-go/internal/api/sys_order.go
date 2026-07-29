package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	orderservice "github.com/earthyzinc/dehaze-go/internal/service/order"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

type OrderApi struct {
	orderService orderservice.IOrderService
}

func NewOrderApi(orderService orderservice.IOrderService) *OrderApi {
	return &OrderApi{orderService: orderService}
}

func (api *OrderApi) Create(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var form bo.OrderCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.orderService.Create(c.Request.Context(), userID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "创建订单成功", c)
}

func (api *OrderApi) ListMy(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	q := &query.MyOrderQuery{
		Status:   c.Query("status"),
		PageNum:  1,
		PageSize: 10,
	}
	if v := c.Query("pageNum"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			q.PageNum = n
		}
	}
	if v := c.Query("pageSize"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			q.PageSize = n
		}
	}

	result, err := api.orderService.ListMy(c.Request.Context(), userID, q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *OrderApi) GetDetail(c *gin.Context) {
	orderNo := c.Param("orderNo")
	if orderNo == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "订单号不能为空"))
		return
	}

	result, err := api.orderService.GetDetail(c.Request.Context(), orderNo)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *OrderApi) Cancel(c *gin.Context) {
	orderNo := c.Param("orderNo")
	if orderNo == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "订单号不能为空"))
		return
	}

	reason := c.Query("reason")
	if reason == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "取消原因不能为空"))
		return
	}

	if err := api.orderService.Cancel(c.Request.Context(), orderNo, reason); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("取消订单成功", c)
}

func (api *OrderApi) Pay(c *gin.Context) {
	orderNo := c.Param("orderNo")
	if orderNo == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "订单号不能为空"))
		return
	}

	var req bo.PayRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.orderService.Pay(c.Request.Context(), orderNo, &req)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "支付成功", c)
}

func (api *OrderApi) ApplyRefund(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	orderNo := c.Param("orderNo")
	if orderNo == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "订单号不能为空"))
		return
	}

	var form bo.RefundApplyForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.orderService.ApplyRefund(c.Request.Context(), userID, orderNo, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("申请退款成功", c)
}

func (api *OrderApi) UpdateAutoRenewConfig(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var form bo.AutoRenewConfigForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.orderService.UpdateAutoRenewConfig(c.Request.Context(), userID, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("更新自动续费配置成功", c)
}

func (api *OrderApi) GetAutoRenewConfig(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	packageID, err := strconv.ParseInt(c.Query("packageId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "packageId 参数不正确"))
		return
	}

	result, err := api.orderService.GetAutoRenewConfig(c.Request.Context(), userID, packageID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *OrderApi) GetPage(c *gin.Context) {
	q := &query.OrderPageQuery{
		OrderNo:       c.Query("orderNo"),
		Keywords:      c.Query("keywords"),
		Status:        c.Query("status"),
		PayMethod:     c.Query("payMethod"),
		PaidTimeStart: c.Query("paidTimeStart"),
		PaidTimeEnd:   c.Query("paidTimeEnd"),
		PageNum:       1,
		PageSize:      10,
	}
	if v := c.Query("pageNum"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			q.PageNum = n
		}
	}
	if v := c.Query("pageSize"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			q.PageSize = n
		}
	}
	if v := c.Query("amountMin"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			q.AmountMin = &n
		}
	}
	if v := c.Query("amountMax"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			q.AmountMax = &n
		}
	}

	result, err := api.orderService.GetPage(c.Request.Context(), q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *OrderApi) ListRefunds(c *gin.Context) {
	q := &query.RefundPageQuery{
		OrderNo:        c.Query("orderNo"),
		Keywords:       c.Query("keywords"),
		Status:         c.Query("status"),
		ApplyTimeStart: c.Query("applyTimeStart"),
		ApplyTimeEnd:   c.Query("applyTimeEnd"),
		PageNum:        1,
		PageSize:       10,
	}
	if v := c.Query("pageNum"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			q.PageNum = n
		}
	}
	if v := c.Query("pageSize"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			q.PageSize = n
		}
	}

	result, err := api.orderService.ListRefunds(c.Request.Context(), q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *OrderApi) ApproveRefund(c *gin.Context) {
	auditorID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	refundID, err := strconv.ParseInt(c.Param("refundId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "退款ID格式不正确"))
		return
	}

	var form bo.RefundAuditForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.orderService.ApproveRefund(c.Request.Context(), auditorID, refundID, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("退款审核通过", c)
}

func (api *OrderApi) RejectRefund(c *gin.Context) {
	auditorID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	refundID, err := strconv.ParseInt(c.Param("refundId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "退款ID格式不正确"))
		return
	}

	var form bo.RefundAuditForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.orderService.RejectRefund(c.Request.Context(), auditorID, refundID, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("退款已驳回", c)
}

func (api *OrderApi) GetStats(c *gin.Context) {
	startTime := c.Query("startTime")
	endTime := c.Query("endTime")

	result, err := api.orderService.GetStats(c.Request.Context(), startTime, endTime)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

package api

import (
	"strconv"
	"strings"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	pkgsaleservice "github.com/earthyzinc/dehaze-go/internal/service/pkgsale"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

type PackageApi struct {
	packageService pkgsaleservice.IPackageService
	couponService  pkgsaleservice.ICouponService
}

func NewPackageApi(packageService pkgsaleservice.IPackageService, couponService pkgsaleservice.ICouponService) *PackageApi {
	return &PackageApi{packageService: packageService, couponService: couponService}
}

func (api *PackageApi) ListOnSale(c *gin.Context) {
	result, err := api.packageService.ListOnSale(c.Request.Context())
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *PackageApi) GetDetail(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "套餐ID格式不正确"))
		return
	}

	result, err := api.packageService.GetDetail(c.Request.Context(), id)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *PackageApi) CalculatePrice(c *gin.Context) {
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

	var userCouponID *int64
	if v := c.Query("userCouponId"); v != "" {
		id, err := strconv.ParseInt(v, 10, 64)
		if err == nil && id > 0 {
			userCouponID = &id
		}
	}

	result, err := api.packageService.CalculatePrice(c.Request.Context(), userID, packageID, userCouponID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *PackageApi) GetPage(c *gin.Context) {
	q := &query.PackagePageQuery{
		Name:      c.Query("name"),
		LevelCode: c.Query("levelCode"),
		Period:    c.Query("period"),
		StartTime: c.Query("startTime"),
		EndTime:   c.Query("endTime"),
		PageNum:   1,
		PageSize:  20,
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
	if v := c.Query("status"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			q.Status = &n
		}
	}

	result, err := api.packageService.GetPage(c.Request.Context(), q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *PackageApi) GetForm(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "套餐ID格式不正确"))
		return
	}

	result, err := api.packageService.GetForm(c.Request.Context(), id)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *PackageApi) Add(c *gin.Context) {
	var form bo.PackageForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.packageService.Create(c.Request.Context(), &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("创建套餐成功", c)
}

func (api *PackageApi) Update(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "套餐ID格式不正确"))
		return
	}

	var form bo.PackageForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.packageService.Update(c.Request.Context(), id, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("更新套餐成功", c)
}

func (api *PackageApi) UpdateStatus(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "套餐ID格式不正确"))
		return
	}

	status, err := strconv.Atoi(c.Query("status"))
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "status 参数不正确"))
		return
	}

	if err := api.packageService.UpdateStatus(c.Request.Context(), id, status); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("更新状态成功", c)
}

func (api *PackageApi) DeleteByIds(c *gin.Context) {
	idsStr := c.Param("ids")
	if idsStr == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "请选择要删除的套餐"))
		return
	}

	parts := strings.Split(idsStr, ",")
	ids := make([]int64, 0, len(parts))
	for _, p := range parts {
		id, err := strconv.ParseInt(strings.TrimSpace(p), 10, 64)
		if err != nil {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "套餐ID格式不正确"))
			return
		}
		ids = append(ids, id)
	}

	if err := api.packageService.DeleteByIDs(c.Request.Context(), ids); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("删除套餐成功", c)
}

func (api *PackageApi) GetSalesStats(c *gin.Context) {
	result, err := api.packageService.GetSalesStats(c.Request.Context())
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

// ============ 优惠券接口 ============

func (api *PackageApi) ListMyCoupons(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var status *int
	if v := c.Query("status"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			status = &n
		}
	}

	result, err := api.couponService.ListMy(c.Request.Context(), userID, status)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *PackageApi) ReceiveCoupon(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	couponID, err := strconv.ParseInt(c.Param("couponId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "优惠券ID格式不正确"))
		return
	}

	result, err := api.couponService.Receive(c.Request.Context(), userID, couponID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "领取成功", c)
}

func (api *PackageApi) GetCouponPage(c *gin.Context) {
	q := &query.CouponPageQuery{
		Name:     c.Query("name"),
		Type:     c.Query("type"),
		PageNum:  1,
		PageSize: 20,
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
	if v := c.Query("status"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			q.Status = &n
		}
	}

	result, err := api.couponService.GetPage(c.Request.Context(), q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *PackageApi) AddCoupon(c *gin.Context) {
	var form bo.CouponForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.couponService.Create(c.Request.Context(), &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "创建优惠券成功", c)
}

func (api *PackageApi) BatchDistributeCoupon(c *gin.Context) {
	var form bo.CouponBatchDistributeForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.couponService.BatchDistribute(c.Request.Context(), &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "批量发放完成", c)
}

func (api *PackageApi) UpdateCoupon(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "优惠券ID格式不正确"))
		return
	}

	var form bo.CouponForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.couponService.Update(c.Request.Context(), id, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("更新优惠券成功", c)
}

func (api *PackageApi) DeleteCouponsByIds(c *gin.Context) {
	idsStr := c.Param("ids")
	if idsStr == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "请选择要删除的优惠券"))
		return
	}

	parts := strings.Split(idsStr, ",")
	ids := make([]int64, 0, len(parts))
	for _, p := range parts {
		id, err := strconv.ParseInt(strings.TrimSpace(p), 10, 64)
		if err != nil {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "优惠券ID格式不正确"))
			return
		}
		ids = append(ids, id)
	}

	if err := api.couponService.DeleteByIDs(c.Request.Context(), ids); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("删除优惠券成功", c)
}

package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	pkgsaleservice "github.com/earthyzinc/dehaze-go/internal/service/pkgsale"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/gin-gonic/gin"
)

type PromotionApi struct {
	promotionService pkgsaleservice.IPromotionService
}

func NewPromotionApi(promotionService pkgsaleservice.IPromotionService) *PromotionApi {
	return &PromotionApi{promotionService: promotionService}
}

func (api *PromotionApi) GetPage(c *gin.Context) {
	// 分页走全局 parsePagination（pageNum>=1、1<=pageSize<=100，越界 A0400），
	// 与 python PromotionQuery(BasePageQuery) 同口径。此前这里对越界值静默回退默认值，
	// 会让 pageSize=500 在 go 返回 500 行、在 python 报 A0400，形成客户端可触发的行为分叉。
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	q := &query.PromotionPageQuery{
		Name:      c.Query("name"),
		Type:      c.Query("type"),
		StartTime: c.Query("startTime"),
		EndTime:   c.Query("endTime"),
		PageNum:   pageNum,
		PageSize:  pageSize,
	}
	if v := c.Query("status"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			q.Status = &n
		}
	}

	result, err := api.promotionService.GetPage(c.Request.Context(), q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *PromotionApi) Add(c *gin.Context) {
	var form bo.PromotionForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.promotionService.Create(c.Request.Context(), &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "创建促销活动成功", c)
}

func (api *PromotionApi) Update(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "促销活动ID格式不正确"))
		return
	}

	var form bo.PromotionForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.promotionService.Update(c.Request.Context(), id, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "更新促销活动成功", c)
}

func (api *PromotionApi) UpdateStatus(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "促销活动ID格式不正确"))
		return
	}

	status, err := strconv.Atoi(c.Query("status"))
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "status 参数不正确"))
		return
	}

	result, err := api.promotionService.UpdateStatus(c.Request.Context(), id, status)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "更新状态成功", c)
}

func (api *PromotionApi) Delete(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "促销活动ID格式不正确"))
		return
	}

	if err := api.promotionService.Delete(c.Request.Context(), id); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("删除促销活动成功", c)
}

func (api *PromotionApi) BindPackages(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "促销活动ID格式不正确"))
		return
	}

	var form bo.PromotionPackageForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.promotionService.BindPackages(c.Request.Context(), id, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("关联套餐成功", c)
}

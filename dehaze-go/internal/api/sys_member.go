package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	memberservice "github.com/earthyzinc/dehaze-go/internal/service/member"
	orderservice "github.com/earthyzinc/dehaze-go/internal/service/order"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

type MemberApi struct {
	memberService memberservice.IMemberService
	orderService  orderservice.IOrderService
}

func NewMemberApi(memberService memberservice.IMemberService, orderService orderservice.IOrderService) *MemberApi {
	return &MemberApi{memberService: memberService, orderService: orderService}
}

func (api *MemberApi) GetProfile(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.memberService.GetProfile(c.Request.Context(), userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

// GetBenefitSummary 当前用户权益概览（GET /members/benefit-summary）
func (api *MemberApi) GetBenefitSummary(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.memberService.GetBenefitSummary(c.Request.Context(), userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

// GetMemberBenefitUsage 管理端查看指定用户权益（GET /members/{userId}/benefit-usage，权限 member:list）。
// python 侧复用同一个 get_benefit_summary，故 payload 与用户端同构。
func (api *MemberApi) GetMemberBenefitUsage(c *gin.Context) {
	userID, err := strconv.ParseInt(c.Param("userId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "用户ID格式不正确"))
		return
	}

	result, err := api.memberService.GetBenefitSummary(c.Request.Context(), userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

// GetTrialStatus 当前用户试用引导状态（GET /members/trial-status）
func (api *MemberApi) GetTrialStatus(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.memberService.GetTrialStatus(c.Request.Context(), userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

// GetMemberOperationLogs 目标会员操作日志（GET /members/{userId}/operation-logs，权限 member:list）。
// 分页口径对齐 python `Query(default=1, ge=1)` / `Query(default=10, ge=1, le=100)`。
func (api *MemberApi) GetMemberOperationLogs(c *gin.Context) {
	userID, err := strconv.ParseInt(c.Param("userId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "用户ID格式不正确"))
		return
	}

	page, size, ok := parsePaginationWithSize(c, 10)
	if !ok {
		return
	}

	result, err := api.memberService.ListMemberAuditLogs(c.Request.Context(), userID, page, size)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *MemberApi) GetGrowthLogs(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	q := &query.GrowthLogQuery{
		ChangeType: c.Query("changeType"),
		StartTime:  c.Query("startTime"),
		EndTime:    c.Query("endTime"),
		PageNum:    pageNum,
		PageSize:   pageSize,
	}

	result, err := api.memberService.ListGrowthLogs(c.Request.Context(), userID, q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *MemberApi) SignIn(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.memberService.SignIn(c.Request.Context(), userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "签到成功", c)
}

func (api *MemberApi) GetSignInCalendar(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	year, err := strconv.Atoi(c.Query("year"))
	if err != nil || year <= 0 {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "year 参数不正确"))
		return
	}
	month, err := strconv.Atoi(c.Query("month"))
	if err != nil || month < 1 || month > 12 {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "month 参数不正确"))
		return
	}

	result, err := api.memberService.GetSignInCalendar(c.Request.Context(), userID, year, month)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *MemberApi) GetPage(c *gin.Context) {
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	q := &query.MemberPageQuery{
		Keywords:        c.Query("keywords"),
		LevelCode:       c.Query("levelCode"),
		ExpireTimeStart: c.Query("expireTimeStart"),
		ExpireTimeEnd:   c.Query("expireTimeEnd"),
		PageNum:         pageNum,
		PageSize:        pageSize,
	}
	if v := c.Query("status"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			q.Status = &n
		}
	}
	if v := c.Query("growthMin"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			q.GrowthMin = &n
		}
	}
	if v := c.Query("growthMax"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			q.GrowthMax = &n
		}
	}

	result, err := api.memberService.ListPagedMembers(c.Request.Context(), q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *MemberApi) GetDetail(c *gin.Context) {
	userID, err := strconv.ParseInt(c.Param("userId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "用户ID格式不正确"))
		return
	}

	// 默认仅本人可见；持 member:list 权限可查任意会员（python get_member_detail 同口径）
	if cUserID, err := security.RequireUserID(c); err == nil && cUserID != userID {
		if ok, _ := security.HasAnyPermission(c, "member:list"); !ok {
			_ = c.Error(common.NewBizError(common.ACCESS_UNAUTHORIZED, "无权查看他人会员详情"))
			return
		}
	}

	result, err := api.memberService.GetMemberDetail(c.Request.Context(), userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *MemberApi) AdjustLevel(c *gin.Context) {
	operatorID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	userID, err := strconv.ParseInt(c.Param("userId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "用户ID格式不正确"))
		return
	}

	var form bo.MemberLevelAdjustForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	// 参数合法后再做权限校验（与 FastAPI body 校验先行顺序对齐）
	if err := middleware.CheckPermission(c, "member:level:edit"); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.memberService.AdjustLevel(c.Request.Context(), userID, operatorID, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("等级调整成功", c)
}

func (api *MemberApi) AdjustGrowth(c *gin.Context) {
	operatorID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	userID, err := strconv.ParseInt(c.Param("userId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "用户ID格式不正确"))
		return
	}

	var form bo.MemberGrowthAdjustForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	// 参数合法后再做权限校验（与 FastAPI body 校验先行顺序对齐）
	if err := middleware.CheckPermission(c, "member:growth:edit"); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.memberService.AdjustGrowth(c.Request.Context(), userID, operatorID, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("成长值调整成功", c)
}

func (api *MemberApi) UpdateStatus(c *gin.Context) {
	userID, err := strconv.ParseInt(c.Param("userId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "用户ID格式不正确"))
		return
	}

	var form bo.MemberStatusForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	// 参数合法后再做权限校验（与 FastAPI body 校验先行顺序对齐）
	if err := middleware.CheckPermission(c, "member:status:edit"); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.memberService.UpdateStatus(c.Request.Context(), userID, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("状态更新成功", c)
}

func (api *MemberApi) ListBenefits(c *gin.Context) {
	result, err := api.memberService.ListBenefits(c.Request.Context())
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *MemberApi) UpdateBenefit(c *gin.Context) {
	levelCode := c.Param("levelCode")
	if levelCode == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "等级编码不能为空"))
		return
	}

	var form bo.BenefitForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.memberService.UpdateBenefit(c.Request.Context(), levelCode, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("权益配置更新成功", c)
}

// GetMemberGrowthLogs 会员成长值流水（管理端按用户查，python get_member_growth_logs 同口径）
func (api *MemberApi) GetMemberGrowthLogs(c *gin.Context) {
	targetUserID, err := strconv.ParseInt(c.Param("userId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "用户ID格式不正确"))
		return
	}

	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	q := &query.GrowthLogQuery{
		ChangeType: c.Query("changeType"),
		StartTime:  c.Query("startTime"),
		EndTime:    c.Query("endTime"),
		PageNum:    pageNum,
		PageSize:   pageSize,
	}

	result, err := api.memberService.ListGrowthLogs(c.Request.Context(), targetUserID, q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

// GetMemberConsumptionRecords 会员消费记录（管理端按用户查，python 复用 order list_my 同口径）
func (api *MemberApi) GetMemberConsumptionRecords(c *gin.Context) {
	targetUserID, err := strconv.ParseInt(c.Param("userId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "用户ID格式不正确"))
		return
	}

	// python member.py:196-197 默认 pageSize 10
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	q := &query.MyOrderQuery{PageNum: pageNum, PageSize: pageSize}
	q.Status = c.Query("status")

	result, err := api.orderService.ListMy(c.Request.Context(), targetUserID, q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

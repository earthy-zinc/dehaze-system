package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	memberservice "github.com/earthyzinc/dehaze-go/internal/service/member"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

type MemberApi struct {
	memberService memberservice.IMemberService
}

func NewMemberApi(memberService memberservice.IMemberService) *MemberApi {
	return &MemberApi{memberService: memberService}
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

func (api *MemberApi) GetGrowthLogs(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	q := &query.GrowthLogQuery{
		ChangeType: c.Query("changeType"),
		StartTime:  c.Query("startTime"),
		EndTime:    c.Query("endTime"),
		PageNum:    1,
		PageSize:   20,
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
	q := &query.MemberPageQuery{
		Keywords:        c.Query("keywords"),
		LevelCode:       c.Query("levelCode"),
		ExpireTimeStart: c.Query("expireTimeStart"),
		ExpireTimeEnd:   c.Query("expireTimeEnd"),
		PageNum:         1,
		PageSize:        20,
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

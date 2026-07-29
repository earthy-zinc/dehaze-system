package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	feedbackservice "github.com/earthyzinc/dehaze-go/internal/service/feedback"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

type FeedbackApi struct {
	ratingService   feedbackservice.IRatingService
	feedbackService feedbackservice.IFeedbackService
}

func NewFeedbackApi(
	ratingService feedbackservice.IRatingService,
	feedbackService feedbackservice.IFeedbackService,
) *FeedbackApi {
	return &FeedbackApi{
		ratingService:   ratingService,
		feedbackService: feedbackService,
	}
}

// ============ 评价接口 ============

func (api *FeedbackApi) CreateRating(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var form bo.RatingCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	id, err := api.ratingService.CreateRating(c.Request.Context(), userID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(gin.H{"id": id}, "评价成功", c)
}

func (api *FeedbackApi) UpdateRating(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}

	var form bo.RatingCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.ratingService.UpdateRating(c.Request.Context(), userID, id, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("修改评价成功", c)
}

func (api *FeedbackApi) ListMyRatings(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	pageNum, pageSize := parsePagination(c)
	result, err := api.ratingService.ListMyRatings(c.Request.Context(), userID, pageNum, pageSize)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *FeedbackApi) GetRatingByPrediction(c *gin.Context) {
	predLogID, err := strconv.ParseInt(c.Param("predictionLogId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "处理记录ID格式不正确"))
		return
	}

	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	result, err := api.ratingService.GetRatingByPrediction(c.Request.Context(), userID, predLogID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *FeedbackApi) ListRatings(c *gin.Context) {
	q := &query.RatingPageQuery{
		Keywords:  c.Query("keywords"),
		StartTime: c.Query("startTime"),
		EndTime:   c.Query("endTime"),
		PageNum:   1,
		PageSize:  10,
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
	if v := c.Query("algorithmId"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			q.AlgorithmID = &n
		}
	}
	if v := c.Query("ratingMin"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			q.RatingMin = &n
		}
	}
	if v := c.Query("ratingMax"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			q.RatingMax = &n
		}
	}
	if v := c.Query("hasComment"); v != "" {
		b := v == "true" || v == "1"
		q.HasComment = &b
	}

	result, err := api.ratingService.ListPagedRatings(c.Request.Context(), q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *FeedbackApi) HideRating(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}

	if err := api.ratingService.HideRating(c.Request.Context(), id); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("隐藏评价成功", c)
}

func (api *FeedbackApi) ReplyRating(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}

	var body struct {
		Content string `json:"content"`
	}
	if err := c.ShouldBindJSON(&body); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.ratingService.ReplyRating(c.Request.Context(), id, body.Content); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("回复评价成功", c)
}

func (api *FeedbackApi) GetRatingStats(c *gin.Context) {
	startTime := c.Query("startTime")
	endTime := c.Query("endTime")

	result, err := api.ratingService.GetRatingStats(c.Request.Context(), startTime, endTime)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

// ============ 反馈接口 ============

func (api *FeedbackApi) CreateFeedback(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var form bo.FeedbackCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	id, err := api.feedbackService.CreateFeedback(c.Request.Context(), userID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(gin.H{"id": id}, "创建反馈成功", c)
}

func (api *FeedbackApi) ListMyFeedback(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	pageNum, pageSize := parsePagination(c)
	result, err := api.feedbackService.ListMyFeedback(c.Request.Context(), userID, pageNum, pageSize)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *FeedbackApi) GetFeedbackDetail(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	isAdmin := security.IsAdmin(c)

	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}

	result, err := api.feedbackService.GetFeedbackDetail(c.Request.Context(), id, userID, isAdmin)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *FeedbackApi) SupplementFeedback(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}

	var form bo.FeedbackSupplementForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.feedbackService.SupplementFeedback(c.Request.Context(), userID, id, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("补充说明成功", c)
}

func (api *FeedbackApi) ListFeedback(c *gin.Context) {
	q := &query.FeedbackPageQuery{
		Keywords:      c.Query("keywords"),
		FeedbackType:  c.Query("feedbackType"),
		Status:        c.Query("status"),
		RelatedModule: c.Query("relatedModule"),
		StartTime:     c.Query("startTime"),
		EndTime:       c.Query("endTime"),
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
	if v := c.Query("priority"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			q.Priority = &n
		}
	}
	if v := c.Query("assigneeId"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			q.AssigneeID = &n
		}
	}

	result, err := api.feedbackService.ListPagedFeedback(c.Request.Context(), q)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func (api *FeedbackApi) AssignFeedback(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}

	var form bo.FeedbackAssignForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.feedbackService.AssignFeedback(c.Request.Context(), id, form.AssigneeID); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("分配处理人成功", c)
}

func (api *FeedbackApi) ReplyFeedback(c *gin.Context) {
	adminID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}

	var form bo.FeedbackReplyForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.feedbackService.ReplyFeedback(c.Request.Context(), adminID, id, &form); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("回复反馈成功", c)
}

func (api *FeedbackApi) CloseFeedback(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}

	var form bo.FeedbackCloseForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.feedbackService.CloseFeedback(c.Request.Context(), id, form.CloseReason); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("关闭反馈成功", c)
}

func (api *FeedbackApi) UpdateFeedbackTags(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}

	var tags []string
	if err := c.ShouldBindJSON(&tags); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.feedbackService.UpdateFeedbackTags(c.Request.Context(), id, tags); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("设置标签成功", c)
}

func (api *FeedbackApi) GetFeedbackStats(c *gin.Context) {
	startTime := c.Query("startTime")
	endTime := c.Query("endTime")

	result, err := api.feedbackService.GetFeedbackStats(c.Request.Context(), startTime, endTime)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

func parsePagination(c *gin.Context) (int, int) {
	pageNum := 1
	pageSize := 10
	if v := c.Query("pageNum"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			pageNum = n
		}
	}
	if v := c.Query("pageSize"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			pageSize = n
		}
	}
	return pageNum, pageSize
}

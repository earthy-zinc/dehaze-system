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

	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
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
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	q := &query.RatingPageQuery{
		Keywords:  c.Query("keywords"),
		StartTime: c.Query("startTime"),
		EndTime:   c.Query("endTime"),
		PageNum:   pageNum,
		PageSize:  pageSize,
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

	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
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
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	q := &query.FeedbackPageQuery{
		Keywords:      c.Query("keywords"),
		FeedbackType:  c.Query("feedbackType"),
		Status:        c.Query("status"),
		RelatedModule: c.Query("relatedModule"),
		StartTime:     c.Query("startTime"),
		EndTime:       c.Query("endTime"),
		PageNum:       pageNum,
		PageSize:      pageSize,
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

// parsePaginationWithSize 解析分页参数，对齐 python 各端点的
// `pageNum: Query(default=1, ge=1)` / `pageSize: Query(default=<N>, ge=1, le=100)`：
// 未传取 defaultSize；传了但非数字、<1 或 >100 一律 A0400，不做静默钳制
// （静默钳制会让 pageSize=500 在 go 返回 500 行、在 python 报 A0400，形成客户端可触发的行为分叉）。
//
// 各端点的 defaultSize 必须逐个对 python router 核实，不可一刀切 10——favorite/message/
// message_template 是 20。它是分页解析的唯一实现，不得再新增第二套。
// parsePaginationNamed 与 parsePaginationWithSize 同语义，但参数名可配——
// python 模型售价端点用的是 `page`/`size`（schema/ai_model_price.py:63），不是 pageNum/pageSize。
func parsePaginationNamed(c *gin.Context, numKey, sizeKey string, defaultSize int) (int, int, bool) {
	rawPageNum, numOK := parseOptionalInt(c.Query(numKey))
	rawPageSize, sizeOK := parseOptionalInt(c.Query(sizeKey))
	pageNum, pageSize := 1, defaultSize
	if rawPageNum != nil {
		pageNum = *rawPageNum
	}
	if rawPageSize != nil {
		pageSize = *rawPageSize
	}
	if !numOK || !sizeOK || pageNum < 1 || pageSize < 1 || pageSize > 100 {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR,
			"分页参数不合法："+numKey+">=1，1<="+sizeKey+"<=100"))
		return 0, 0, false
	}
	return pageNum, pageSize, true
}

// parsePaginationWithSize pageNum/pageSize 端点的简写。
func parsePaginationWithSize(c *gin.Context, defaultSize int) (int, int, bool) {
	return parsePaginationNamed(c, "pageNum", "pageSize", defaultSize)
}

// parsePagination defaultSize=10 端点的简写（python BasePageQuery 主口径）。
func parsePagination(c *gin.Context) (int, int, bool) {
	return parsePaginationWithSize(c, 10)
}

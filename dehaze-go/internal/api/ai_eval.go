package api

import (
	"strconv"

	aidomain "github.com/earthyzinc/dehaze-go/internal/service/aidomain"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

// AiEvalApi 智能体评测集/样本与评测中心（A 类查询与配置）。
type AiEvalApi struct {
	eval   *aidomain.EvalService
	center *aidomain.EvalCenterService
}

// NewAiEvalApi 构造 AiEvalApi。
func NewAiEvalApi(eval *aidomain.EvalService, center *aidomain.EvalCenterService) *AiEvalApi {
	return &AiEvalApi{eval: eval, center: center}
}

// CreateEvalDataset 创建评测集。
func (a *AiEvalApi) CreateEvalDataset(c *gin.Context) {
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	var form aidomain.EvalDatasetCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.eval.CreateDataset(c.Request.Context(), id, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListEvalDatasets 评测集列表。
func (a *AiEvalApi) ListEvalDatasets(c *gin.Context) {
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.eval.ListDatasets(c.Request.Context(), id)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdateEvalDataset 更新评测集。
func (a *AiEvalApi) UpdateEvalDataset(c *gin.Context) {
	id, datasetID, ok := parseAgentAndDataset(c)
	if !ok {
		return
	}
	var form aidomain.EvalDatasetUpdateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.eval.UpdateDataset(c.Request.Context(), id, datasetID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// DeleteEvalDataset 删除评测集。
func (a *AiEvalApi) DeleteEvalDataset(c *gin.Context) {
	id, datasetID, ok := parseAgentAndDataset(c)
	if !ok {
		return
	}
	operatorID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	if err := a.eval.DeleteDataset(c.Request.Context(), id, datasetID, operatorID); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("一切ok", c)
}

// CreateEvalSample 创建评测样本。
func (a *AiEvalApi) CreateEvalSample(c *gin.Context) {
	id, datasetID, ok := parseAgentAndDataset(c)
	if !ok {
		return
	}
	var form aidomain.EvalSampleCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.eval.CreateSample(c.Request.Context(), id, datasetID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListEvalSamples 评测样本列表。
func (a *AiEvalApi) ListEvalSamples(c *gin.Context) {
	id, datasetID, ok := parseAgentAndDataset(c)
	if !ok {
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.eval.ListSamples(c.Request.Context(), id, datasetID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdateEvalSample 更新评测样本。
func (a *AiEvalApi) UpdateEvalSample(c *gin.Context) {
	id, sampleID, ok := parseAgentAndSample(c)
	if !ok {
		return
	}
	var form aidomain.EvalSampleUpdateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.eval.UpdateSample(c.Request.Context(), id, sampleID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// DeleteEvalSample 删除评测样本。
func (a *AiEvalApi) DeleteEvalSample(c *gin.Context) {
	id, sampleID, ok := parseAgentAndSample(c)
	if !ok {
		return
	}
	operatorID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	if err := a.eval.DeleteSample(c.Request.Context(), id, sampleID, operatorID); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithMessage("一切ok", c)
}

// EvalOverview 评测总览。
func (a *AiEvalApi) EvalOverview(c *gin.Context) {
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.center.Overview(c.Request.Context())
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// EvalTrends 评测历史趋势。
func (a *AiEvalApi) EvalTrends(c *gin.Context) {
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	var agentID *int64
	if raw := c.Query("agentId"); raw != "" {
		parsed, err := strconv.ParseInt(raw, 10, 64)
		if err != nil {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "agentId 取值非法"))
			return
		}
		agentID = &parsed
	}
	// limit 取值范围与 python Query(default=100, ge=1, le=500) 一致：越界报参数错误而非静默取默认
	limit := 100
	if raw := c.Query("limit"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 1 || parsed > 500 {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "limit 取值范围为 1~500"))
			return
		}
		limit = parsed
	}
	start, end, ok := queryTimeRange(c, "startTime", "endTime")
	if !ok {
		return
	}
	result, err := a.center.Trends(c.Request.Context(), agentID, start, end, limit)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// EvalRunCompare 两次评测 run 得分对比。
func (a *AiEvalApi) EvalRunCompare(c *gin.Context) {
	runID, ok := parseID(c, "id")
	if !ok {
		return
	}
	baseRunID, err := strconv.ParseInt(c.Query("baseRunId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "baseRunId 取值非法"))
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.center.CompareRuns(c.Request.Context(), runID, baseRunID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// EvalJudgeStatus 判分模型状态。
func (a *AiEvalApi) EvalJudgeStatus(c *gin.Context) {
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.center.JudgeStatus(c.Request.Context())
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// EvalReviews 人工复核队列。
func (a *AiEvalApi) EvalReviews(c *gin.Context) {
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	status, ok := parseRangedOptionalInt(c.Query("status"), 1, 2)
	if !ok {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "status 仅支持 1(待复核)/2(已复核)"))
		return
	}
	result, err := a.center.ListReviews(c.Request.Context(), status)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// EvalReviewDetail 复核详情。
func (a *AiEvalApi) EvalReviewDetail(c *gin.Context) {
	runID, ok := parseID(c, "id")
	if !ok {
		return
	}
	sampleID, err := strconv.ParseInt(c.Param("sampleId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "sampleId 取值非法"))
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.center.ReviewDetail(c.Request.Context(), runID, sampleID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// SubmitEvalReview 复核结果回填。
func (a *AiEvalApi) SubmitEvalReview(c *gin.Context) {
	id, ok := parseID(c, "id")
	if !ok {
		return
	}
	var form aidomain.EvalReviewSubmitForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	reviewerID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := middleware.CheckPermission(c, agentManagePermission); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.center.SubmitReview(c.Request.Context(), id, &form, reviewerID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

func parseAgentAndDataset(c *gin.Context) (int64, int64, bool) {
	agentID, ok := parseID(c, "id")
	if !ok {
		return 0, 0, false
	}
	datasetID, err := strconv.ParseInt(c.Param("datasetId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "datasetId 取值非法"))
		return 0, 0, false
	}
	return agentID, datasetID, true
}

func parseAgentAndSample(c *gin.Context) (int64, int64, bool) {
	agentID, ok := parseID(c, "id")
	if !ok {
		return 0, 0, false
	}
	sampleID, err := strconv.ParseInt(c.Param("sampleId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "sampleId 取值非法"))
		return 0, 0, false
	}
	return agentID, sampleID, true
}

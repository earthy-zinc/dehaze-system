package api

import (
	"strconv"

	evaluationservice "github.com/earthyzinc/dehaze-go/internal/service/evaluation"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

// SysEvaluationApi 效果评估 API
type SysEvaluationApi struct {
	service *evaluationservice.EvaluationService
}

func NewSysEvaluationApi(service *evaluationservice.EvaluationService) *SysEvaluationApi {
	return &SysEvaluationApi{service: service}
}

// Evaluate 执行效果评估
func (api *SysEvaluationApi) Evaluate(c *gin.Context) {
	ctx := c.Request.Context()
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var req struct {
		ModelID string `json:"modelId" binding:"required"`
		PredURL string `json:"predUrl" binding:"required"`
		GtURL   string `json:"gtUrl" binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.service.Evaluate(ctx, req.ModelID, req.PredURL, req.GtURL, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetEvaluationLog 查询评估任务状态（仅任务归属用户可访问）
func (api *SysEvaluationApi) GetEvaluationLog(c *gin.Context) {
	ctx := c.Request.Context()
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}

	result, err := api.service.GetTaskStatus(ctx, id, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetMetrics 获取评估指标历史（当前用户，仅已完成状态）
func (api *SysEvaluationApi) GetMetrics(c *gin.Context) {
	ctx := c.Request.Context()
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	modelID := c.Query("modelId")
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}

	result, err := api.service.GetUserMetricsPage(ctx, userID, modelID, pageNum, pageSize)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

// ListEvaluationLogs 评估日志列表（当前用户）
func (api *SysEvaluationApi) ListEvaluationLogs(c *gin.Context) {
	ctx := c.Request.Context()
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	modelID := c.Query("modelId")
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}

	result, err := api.service.GetLogPage(ctx, modelID, userID, pageNum, pageSize)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

package api

import (
	"strconv"

	evaluationservice "github.com/earthyzinc/dehaze-go/internal/service/evaluation"
	"github.com/earthyzinc/dehaze-go/pkg/common"
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
	userID := getCurrentUserID(c)

	var req struct {
		AlgorithmID int64  `json:"algorithmId" binding:"required"`
		PredURL     string `json:"predUrl" binding:"required"`
		GtURL       string `json:"gtUrl" binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.service.Evaluate(ctx, req.AlgorithmID, req.PredURL, req.GtURL, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetEvaluationLog 查询评估任务状态
func (api *SysEvaluationApi) GetEvaluationLog(c *gin.Context) {
	ctx := c.Request.Context()
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}

	log, err := api.service.GetLogByID(ctx, id)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(log, c)
}

// ListEvaluationLogs 评估日志列表
func (api *SysEvaluationApi) ListEvaluationLogs(c *gin.Context) {
	ctx := c.Request.Context()
	algorithmID, _ := strconv.ParseInt(c.Query("algorithmId"), 10, 64)
	pageNum, pageSize := getPageParams(c)

	result, err := api.service.GetLogPage(ctx, algorithmID, pageNum, pageSize)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

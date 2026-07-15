package api

import (
	"strconv"

	predservice "github.com/earthyzinc/dehaze-go/internal/service/prediction"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

// SysPredictionApi 去雾预测 API
type SysPredictionApi struct {
	service *predservice.PredictionService
}

func NewSysPredictionApi(service *predservice.PredictionService) *SysPredictionApi {
	return &SysPredictionApi{service: service}
}

// Predict 执行去雾预测
func (api *SysPredictionApi) Predict(c *gin.Context) {
	ctx := c.Request.Context()
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var req struct {
		AlgorithmID int64  `json:"algorithmId" binding:"required"`
		ImageURL    string `json:"imageUrl" binding:"required"`
		Params      string `json:"params"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.service.Predict(ctx, req.AlgorithmID, req.ImageURL, req.Params, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetPredictionLog 查询预测任务状态
func (api *SysPredictionApi) GetPredictionLog(c *gin.Context) {
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

// ListPredictionLogs 预测日志列表
func (api *SysPredictionApi) ListPredictionLogs(c *gin.Context) {
	ctx := c.Request.Context()
	var algorithmID int64
	if algorithmIDStr := c.Query("algorithmId"); algorithmIDStr != "" {
		var err error
		algorithmID, err = strconv.ParseInt(algorithmIDStr, 10, 64)
		if err != nil {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "algorithmId格式不正确"))
			return
		}
	}
	pageNum, pageSize := getPageParams(c)

	result, err := api.service.GetLogPage(ctx, algorithmID, pageNum, pageSize)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

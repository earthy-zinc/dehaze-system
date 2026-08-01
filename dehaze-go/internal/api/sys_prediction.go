package api

import (
	"strconv"

	fileservice "github.com/earthyzinc/dehaze-go/internal/service/file"
	predservice "github.com/earthyzinc/dehaze-go/internal/service/prediction"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

// SysPredictionApi 去雾预测 API
type SysPredictionApi struct {
	service     *predservice.PredictionService
	fileService *fileservice.FileService
}

func NewSysPredictionApi(service *predservice.PredictionService, fileService *fileservice.FileService) *SysPredictionApi {
	return &SysPredictionApi{service: service, fileService: fileService}
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
		ImageURL    string `json:"imageUrl"`
		FileID      *int64 `json:"fileId"`
		Params      string `json:"params"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	// fileId 优先：用文件运行时拼接的 URL（对齐 Java/Python 的 resolveImageUrl）
	imageURL := req.ImageURL
	if req.FileID != nil {
		file, err := api.fileService.GetFileById(ctx, *req.FileID)
		if err != nil {
			_ = c.Error(err)
			return
		}
		imageURL = api.fileService.GetURL(ctx, &file)
	}
	if imageURL == "" {
		_ = c.Error(common.NewBizError(common.PARAM_IS_NULL, "图片来源不能为空，请提供 fileId 或 imageUrl"))
		return
	}

	result, err := api.service.Predict(ctx, req.AlgorithmID, imageURL, req.Params, userID)
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

	result, err := api.service.GetTaskStatus(ctx, id)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
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

// BatchPredict 批量去雾预测
func (api *SysPredictionApi) BatchPredict(c *gin.Context) {
	ctx := c.Request.Context()
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var req struct {
		AlgorithmID int64                     `json:"algorithmId" binding:"required"`
		Items       []predservice.BatchPredictionInput `json:"items" binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	results, err := api.service.BatchPredict(ctx, req.AlgorithmID, req.Items, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(struct {
		Total   int                            `json:"total"`
		Results []predservice.PredictionResult `json:"results"`
	}{
		Total:   len(results),
		Results: results,
	}, c)
}

// GetQuota 查询剩余处理次数
func (api *SysPredictionApi) GetQuota(c *gin.Context) {
	ctx := c.Request.Context()
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.service.GetQuota(ctx, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

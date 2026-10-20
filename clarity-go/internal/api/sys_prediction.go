package api

import (
	"strconv"

	fileservice "github.com/earthyzinc/clarity-go/internal/service/file"
	predservice "github.com/earthyzinc/clarity-go/internal/service/prediction"
	"github.com/earthyzinc/clarity-go/pkg/common"
	"github.com/earthyzinc/clarity-go/pkg/security"
	"github.com/gin-gonic/gin"
)

// SysPredictionApi 图像任务 API（/api/v1/images/*）
type SysPredictionApi struct {
	service     *predservice.PredictionService
	fileService *fileservice.FileService
}

func NewSysPredictionApi(service *predservice.PredictionService, fileService *fileservice.FileService) *SysPredictionApi {
	return &SysPredictionApi{service: service, fileService: fileService}
}

// SubmitImageEdit 提交图像编辑任务（POST /images/edits）
func (api *SysPredictionApi) SubmitImageEdit(c *gin.Context) {
	ctx := c.Request.Context()
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var req struct {
		Model         string `json:"model"`
		ImageURL      string `json:"imageUrl"`
		FileID        *int64 `json:"fileId"`
		Params        string `json:"params"`
		RecommendedBy *int64 `json:"recommendedBy"`
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

	result, err := api.service.Predict(ctx, req.Model, imageURL, req.Params, userID, req.RecommendedBy)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetImageTask 查询图像任务状态（GET /images/tasks/{taskId}）
func (api *SysPredictionApi) GetImageTask(c *gin.Context) {
	ctx := c.Request.Context()
	id, err := strconv.ParseInt(c.Param("taskId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "无效的任务ID: "+c.Param("taskId")))
		return
	}
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.service.GetTaskStatus(ctx, id, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// CancelImageTask 取消图像任务（POST /images/tasks/{taskId}/cancel，幂等）
func (api *SysPredictionApi) CancelImageTask(c *gin.Context) {
	ctx := c.Request.Context()
	taskID, err := strconv.ParseInt(c.Param("taskId"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "无效的任务ID: "+c.Param("taskId")))
		return
	}
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.service.CancelTask(ctx, taskID, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "取消成功", c)
}

// ListImageTasks 图像任务历史列表（GET /images/tasks）
func (api *SysPredictionApi) ListImageTasks(c *gin.Context) {
	ctx := c.Request.Context()
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.service.GetLogPage(ctx, c.Query("model"), userID, pageNum, pageSize)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

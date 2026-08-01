package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/service/compare"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

// CompareApi 效果对比 API
type CompareApi struct {
	service *compare.CompareService
}

func NewCompareApi(service *compare.CompareService) *CompareApi {
	return &CompareApi{service: service}
}

// GenerateReport 生成对比报告（异步任务）
func (api *CompareApi) GenerateReport(c *gin.Context) {
	ctx := c.Request.Context()
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var form compare.CompareReportForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.service.GenerateReport(ctx, userID, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// GetOrDownloadReport 查询报告状态 / 下载报告
// 当 download=true 时返回HTML文件流，否则返回JSON状态
func (api *CompareApi) GetOrDownloadReport(c *gin.Context) {
	ctx := c.Request.Context()
	taskIDStr := c.Param("taskId")
	taskID, err := strconv.ParseInt(taskIDStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "taskId格式不正确"))
		return
	}

	download := c.Query("download") == "true"

	if !download {
		result, err := api.service.GetReportTaskStatus(ctx, taskID)
		if err != nil {
			_ = c.Error(err)
			return
		}
		common.OkWithData(result, c)
		return
	}

	html, err := api.service.GetReportHTML(ctx, taskID)
	if err != nil {
		_ = c.Error(err)
		return
	}

	c.Header("Content-Disposition", "inline; filename=compare-report.html")
	c.Header("Content-Type", "text/html; charset=utf-8")
	c.String(200, html)
}

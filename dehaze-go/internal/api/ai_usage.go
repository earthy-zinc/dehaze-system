package api

import (
	aidomain "github.com/earthyzinc/dehaze-go/internal/service/aidomain"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

// AiUsageApi AI 供应商/模型运营统计。
type AiUsageApi struct {
	usage *aidomain.UsageService
}

// NewAiUsageApi 构造 AiUsageApi。
func NewAiUsageApi(usage *aidomain.UsageService) *AiUsageApi {
	return &AiUsageApi{usage: usage}
}

// GetUsageStats 运营统计（供应商健康看板/模型用量分布/降级与故障）。
func (a *AiUsageApi) GetUsageStats(c *gin.Context) {
	if err := middleware.CheckPermission(c, "ai:model:manage"); err != nil {
		_ = c.Error(err)
		return
	}
	start, end, ok := queryTimeRange(c, "startTime", "endTime")
	if !ok {
		return
	}
	result, err := a.usage.GetUsageStats(c.Request.Context(), start, end)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

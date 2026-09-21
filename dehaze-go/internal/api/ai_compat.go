package api

import (
	"strconv"

	aiservice "github.com/earthyzinc/dehaze-go/internal/service/ai"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

// AiCompatAuditApi AI 兼容 API 调用审计查询（登录用户仅可查本人）
type AiCompatAuditApi struct {
	service *aiservice.CompatAuditService
}

func NewAiCompatAuditApi(service *aiservice.CompatAuditService) *AiCompatAuditApi {
	return &AiCompatAuditApi{service: service}
}

// ListCalls 兼容调用审计分页查询（page/size 与其他分页口径不同，与 python 一致）
func (a *AiCompatAuditApi) ListCalls(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	page, _ := parseAiQueryInt(c, "page", 1)
	size, _ := parseAiQueryInt(c, "size", 20)

	var keyID *int64
	if raw := c.Query("keyId"); raw != "" {
		value, convErr := strconv.ParseInt(raw, 10, 64)
		if convErr != nil {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "keyId 格式不正确"))
			return
		}
		keyID = &value
	}

	result, err := a.service.ListCalls(
		c.Request.Context(),
		userID,
		keyID,
		c.Query("model"),
		c.Query("startTime"),
		c.Query("endTime"),
		page,
		size,
	)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

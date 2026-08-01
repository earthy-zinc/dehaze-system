package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	recservice "github.com/earthyzinc/dehaze-go/internal/service/recommendation"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

type RecommendationApi struct {
	service recservice.IRecommendationService
}

func NewRecommendationApi(service recservice.IRecommendationService) *RecommendationApi {
	return &RecommendationApi{service: service}
}

// Analyze 图像特征分析
func (api *RecommendationApi) Analyze(c *gin.Context) {
	_, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var form bo.AnalyzeForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	// imageId 暂不支持
	if form.ImageID != nil && *form.ImageID > 0 {
		_ = c.Error(common.NewBizError(common.RESOURCE_NOT_FOUND, "imageId方式暂不支持，请使用imageUrl"))
		return
	}
	if form.ImageURL == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "imageId和imageUrl至少提供一个"))
		return
	}

	result, err := api.service.Analyze(c.Request.Context(), &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "分析成功", c)
}

// GetAlgorithmRecommendations 获取算法推荐
func (api *RecommendationApi) GetAlgorithmRecommendations(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var analysisID *int64
	if v := c.Query("analysisId"); v != "" {
		if n, parseErr := strconv.ParseInt(v, 10, 64); parseErr == nil && n > 0 {
			analysisID = &n
		}
	}
	imageMd5 := c.Query("imageMd5")

	result, err := api.service.GetAlgorithmRecommendations(c.Request.Context(), userID, analysisID, imageMd5)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

// SubmitFeedback 推荐反馈
func (api *RecommendationApi) SubmitFeedback(c *gin.Context) {
	_, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var form bo.FeedbackForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	id, err := api.service.SubmitFeedback(c.Request.Context(), &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(gin.H{"id": id}, "反馈成功", c)
}

// GetRules 获取推荐规则（管理员）
func (api *RecommendationApi) GetRules(c *gin.Context) {
	result, err := api.service.GetRules(c.Request.Context())
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

// UpdateRule 更新/新增推荐规则（管理员）
func (api *RecommendationApi) UpdateRule(c *gin.Context) {
	idStr := c.Query("id")
	id := int64(0)
	if idStr != "" {
		var parseErr error
		id, parseErr = strconv.ParseInt(idStr, 10, 64)
		if parseErr != nil {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "id参数格式不正确"))
			return
		}
	}

	var form bo.RuleForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	newID, err := api.service.UpdateRule(c.Request.Context(), id, &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(newID, "操作成功", c)
}

// GetReport 推荐效果报表（管理员）
func (api *RecommendationApi) GetReport(c *gin.Context) {
	startDate := c.Query("startDate")
	endDate := c.Query("endDate")

	result, err := api.service.GetReport(c.Request.Context(), startDate, endDate)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

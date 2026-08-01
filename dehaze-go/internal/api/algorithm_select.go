package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	selectservice "github.com/earthyzinc/dehaze-go/internal/service/algorithm_select"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

// AlgorithmSelectApi 算法选择模块 API
type AlgorithmSelectApi struct {
	service selectservice.IAlgorithmSelectService
}

// NewAlgorithmSelectApi 创建算法选择 API 实例
func NewAlgorithmSelectApi(service selectservice.IAlgorithmSelectService) *AlgorithmSelectApi {
	return &AlgorithmSelectApi{service: service}
}

// GetTree 获取算法选择树（仅已发布算法）
func (api *AlgorithmSelectApi) GetTree(c *gin.Context) {
	tree, err := api.service.GetTree(c.Request.Context())
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(tree, "查询成功", c)
}

// GetDetail 获取算法详情（含样例效果图/评分/使用次数）
func (api *AlgorithmSelectApi) GetDetail(c *gin.Context) {
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}

	detail, err := api.service.GetDetail(c.Request.Context(), id)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(detail, "查询成功", c)
}

// Search 搜索算法（关键词/拼音/标签）
func (api *AlgorithmSelectApi) Search(c *gin.Context) {
	keyword := c.Query("keyword")
	pageNum, pageSize := parsePagination(c)

	result, err := api.service.Search(c.Request.Context(), keyword, pageNum, pageSize)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

// Test 上传图片测试算法效果
func (api *AlgorithmSelectApi) Test(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "ID格式不正确"))
		return
	}

	var body struct {
		ImageURL string `json:"imageUrl" binding:"required"`
	}
	if err := c.ShouldBindJSON(&body); err != nil {
		_ = c.Error(err)
		return
	}

	logID, status, err := api.service.Test(c.Request.Context(), id, body.ImageURL, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(gin.H{"logId": logID, "status": status}, "测试提交成功", c)
}

// Compare 算法对比（最多3个）
func (api *AlgorithmSelectApi) Compare(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var form bo.AlgorithmCompareForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	results, err := api.service.Compare(c.Request.Context(), &form, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(vo.PageResult[vo.AlgorithmCompareVO]{List: results, Total: int64(len(results))}, "对比完成", c)
}

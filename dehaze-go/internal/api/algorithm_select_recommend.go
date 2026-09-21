package api

import (
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/gin-gonic/gin"
)

// Recommend 算法推荐匹配（POST /algorithms/select/recommend，F-M03-007）
func (api *AlgorithmSelectApi) Recommend(c *gin.Context) {
	var form bo.AlgorithmRecommendForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := api.service.Recommend(c.Request.Context(), &form)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "查询成功", c)
}

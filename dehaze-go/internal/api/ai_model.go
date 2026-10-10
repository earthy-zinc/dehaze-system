package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	aiservice "github.com/earthyzinc/dehaze-go/internal/service/ai"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

// AiModelApi AI 模型注册表与用户售价管理接口
type AiModelApi struct {
	service *aiservice.ModelService
}

func NewAiModelApi(service *aiservice.ModelService) *AiModelApi {
	return &AiModelApi{service: service}
}

// ListModels 模型分页列表
func (a *AiModelApi) ListModels(c *gin.Context) {
	// 分页字段（AiPageQuery）已 form:"-"，分页只由本 helper 解析：非数字/越界一律 A0400，
	// 对齐 python AiModelPageQuery(BasePageQuery)。
	pageNum, pageSize, ok := parsePaginationWithSize(c, 10)
	if !ok {
		return
	}
	var query bo.AiModelQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		_ = c.Error(err)
		return
	}
	query.PageNum, query.PageSize = pageNum, pageSize
	result, err := a.service.ListModels(c.Request.Context(), &query)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// ListEnabledModels 启用模型列表（登录用户，按 VIP 等级 + modelType/taskTags 过滤）
func (a *AiModelApi) ListEnabledModels(c *gin.Context) {
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.service.ListEnabledModels(
		c.Request.Context(), userID, c.Query("modelType"), c.Query("taskTags"))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// CreateModel 新增模型
func (a *AiModelApi) CreateModel(c *gin.Context) {
	var form bo.AiModelCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.service.CreateModel(c.Request.Context(), &form, security.GetUserID(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdateModel 更新模型（query 参数 modelId 为业务键）
func (a *AiModelApi) UpdateModel(c *gin.Context) {
	modelID, ok := requireQueryModelID(c)
	if !ok {
		return
	}
	var form bo.AiModelUpdateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.service.UpdateModel(c.Request.Context(), modelID, &form, security.GetUserID(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// DeleteModel 删除模型（query 参数 modelId 为业务键）
func (a *AiModelApi) DeleteModel(c *gin.Context) {
	modelID, ok := requireQueryModelID(c)
	if !ok {
		return
	}
	if err := a.service.DeleteModel(c.Request.Context(), modelID, security.GetUserID(c)); err != nil {
		_ = c.Error(err)
		return
	}
	common.Ok(c)
}

// ListModelPrices 模型用户售价版本分页列表（query 参数 modelId 为业务键）
func (a *AiModelApi) ListModelPrices(c *gin.Context) {
	modelID, ok := requireQueryModelID(c)
	if !ok {
		return
	}
	// python 该端点的参数名是 page/size（schema/ai_model_price.py 的 ModelPriceQuery），
	// 不是 pageNum/pageSize，故用带参数名的变体；校验同样必须先于绑定。
	page, size, ok := parsePaginationNamed(c, "page", "size", 20)
	if !ok {
		return
	}
	var query bo.ModelPriceQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		_ = c.Error(err)
		return
	}
	query.Page, query.Size = page, size
	result, err := a.service.ListModelPrices(c.Request.Context(), modelID, &query)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// CreateModelPrice 新增模型用户售价版本（价格表单内含 modelId）
func (a *AiModelApi) CreateModelPrice(c *gin.Context) {
	var form bo.ModelPriceCreateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}
	result, err := a.service.CreateModelPrice(c.Request.Context(), form.ModelID, &form, security.GetUserID(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// UpdateModelPrice 更新模型用户售价版本（路径参数 priceId）
func (a *AiModelApi) UpdateModelPrice(c *gin.Context) {
	priceID, err := parseAiPathID(c, "priceId")
	if err != nil {
		_ = c.Error(err)
		return
	}
	var form bo.ModelPriceUpdateForm
	if bindErr := c.ShouldBindJSON(&form); bindErr != nil {
		_ = c.Error(bindErr)
		return
	}
	result, err := a.service.UpdateModelPrice(c.Request.Context(), priceID, &form, security.GetUserID(c))
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(result, c)
}

// DeleteModelPrice 删除模型用户售价版本（路径参数 priceId）
func (a *AiModelApi) DeleteModelPrice(c *gin.Context) {
	priceID, err := parseAiPathID(c, "priceId")
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := a.service.DeleteModelPrice(c.Request.Context(), priceID); err != nil {
		_ = c.Error(err)
		return
	}
	common.Ok(c)
}

// requireQueryModelID 读取必填 query 参数 modelId（含 `/` 的厂商业务键不能作路径段）
func requireQueryModelID(c *gin.Context) (string, bool) {
	modelID := c.Query("modelId")
	if modelID == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "缺少参数: modelId"))
		return "", false
	}
	return modelID, true
}

// parsePathID 解析整型路径参数，非法时返回参数错误
func parseAiPathID(c *gin.Context, name string) (int64, error) {
	value, err := strconv.ParseInt(c.Param(name), 10, 64)
	if err != nil {
		return 0, common.NewBizError(common.PARAM_ERROR, "无效的路径参数: "+name)
	}
	return value, nil
}

// lastPathID 取路径上最后一个数字参数（同一路径可含多个 :id，gin 的 Param() 只返回首个匹配）
func lastAiPathID(c *gin.Context) (int64, error) {
	if len(c.Params) == 0 {
		return 0, common.NewBizError(common.PARAM_ERROR, "缺少路径参数")
	}
	raw := c.Params[len(c.Params)-1].Value
	value, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		return 0, common.NewBizError(common.PARAM_ERROR, "无效的路径参数")
	}
	return value, nil
}

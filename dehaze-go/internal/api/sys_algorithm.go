package api

import (
	"strconv"
	"strings"

	"github.com/earthyzinc/dehaze-go/internal/container"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/service"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/gin-gonic/gin"
)

type AlgorithmApi struct {
	algorithmService service.IAlgorithmService
}

// InitServices 初始化服务依赖
func (api *AlgorithmApi) InitServices(container *container.Container) {
	api.algorithmService = container.AlgorithmService()
}

// GetList 获取算法树形表格
func (api *AlgorithmApi) GetList(c *gin.Context) {
	ctx := c.Request.Context()
	var queryParams query.AlgorithmQuery
	_ = c.ShouldBindQuery(&queryParams)

	algorithms, err := api.algorithmService.GetPage(ctx, &queryParams)
	if err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	common.OkWithData(algorithms, c)
}

// GetOptions 获取模型下拉选项列表
func (api *AlgorithmApi) GetOptions(c *gin.Context) {
	ctx := c.Request.Context()
	options, err := api.algorithmService.GetOptions(ctx)
	if err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	common.OkWithData(options, c)
}

// GetById 根据ID获取算法信息
func (api *AlgorithmApi) GetById(c *gin.Context) {
	ctx := c.Request.Context()
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		common.FailWithMessage("参数错误", c)
		return
	}

	form, err := api.algorithmService.GetFormData(ctx, id)
	if err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	common.OkWithData(form, c)
}

// Add 新增算法
func (api *AlgorithmApi) Add(c *gin.Context) {
	ctx := c.Request.Context()
	var algorithmForm bo.AlgorithmFormBO
	_ = c.ShouldBindJSON(&algorithmForm)

	if algorithmForm.ParentID < 0 {
		common.FailWithMessage("父算法ID不能为负数", c)
		return
	}

	err := api.algorithmService.Create(ctx, &algorithmForm)
	if err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	common.OkWithMessage("添加成功", c)
}

// Update 修改算法
func (api *AlgorithmApi) Update(c *gin.Context) {
	ctx := c.Request.Context()
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		common.FailWithMessage("参数错误", c)
		return
	}

	var algorithmForm bo.AlgorithmFormBO
	_ = c.ShouldBindJSON(&algorithmForm)
	algorithmForm.ID = id

	if algorithmForm.ParentID < 0 {
		common.FailWithMessage("父算法ID不能为负数", c)
		return
	}

	err = api.algorithmService.Update(ctx, id, &algorithmForm)
	if err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	common.OkWithMessage("修改成功", c)
}

// Delete 删除算法
func (api *AlgorithmApi) Delete(c *gin.Context) {
	ctx := c.Request.Context()
	idsStr := c.Param("ids")
	if idsStr == "" {
		common.FailWithMessage("参数错误", c)
		return
	}

	idsSlice := []int64{}
	// 解析ID列表
	for _, idStr := range strings.Split(idsStr, ",") {
		if id, err := strconv.ParseInt(idStr, 10, 64); err == nil {
			idsSlice = append(idsSlice, id)
		}
	}

	if len(idsSlice) == 0 {
		common.FailWithMessage("参数错误", c)
		return
	}

	err := api.algorithmService.Delete(ctx, idsSlice)
	if err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	common.OkWithMessage("删除成功", c)
}

// UpdateStatus 更新算法状态
func (api *AlgorithmApi) UpdateStatus(c *gin.Context) {
	ctx := c.Request.Context()
	id, err := strconv.ParseInt(c.Param("id"), 10, 64)
	if err != nil {
		common.FailWithMessage("参数错误", c)
		return
	}

	type StatusRequest struct {
		Status int8 `json:"status"`
	}
	var req StatusRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		common.FailWithMessage("参数错误", c)
		return
	}

	err = api.algorithmService.UpdateStatus(ctx, id, req.Status)
	if err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	common.OkWithMessage("状态更新成功", c)
}

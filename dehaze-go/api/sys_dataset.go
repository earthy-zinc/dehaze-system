package api

import (
	"strconv"
	"strings"

	"github.com/earthyzinc/dehaze-go/common"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/gin-gonic/gin"
)

type SysDatasetApi struct {
	datasetService service.DatasetService
}

// GetDatasetList 数据集树形列表
// @Summary 数据集树形列表
// @Description 获取数据集树形列表
// @Tags 数据集接口
// @Accept application/json
// @Produce application/json
// @Param keywords query string false "关键字"
// @Success 200 {object} vo.Result{data=[]vo.DatasetVO}
// @Router /api/v1/dataset [get]
func (api *SysDatasetApi) GetDatasetList(c *gin.Context) {
	// 解析查询参数
	var queryParams query.DatasetQuery
	queryParams.Keywords = c.Query("keywords")

	// 调用服务获取数据集列表
	datasetVOs, err := api.datasetService.GetDatasetList(queryParams)
	if err != nil {
		common.FailWithMessage("获取数据集列表失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(datasetVOs, "查询成功", c)
}

// GetDatasetOptions 数据集下拉选项
// @Summary 数据集下拉选项
// @Description 获取数据集下拉选项
// @Tags 数据集接口
// @Accept application/json
// @Produce application/json
// @Success 200 {object} vo.Result{data=[]vo.Option}
// @Router /api/v1/dataset/options [get]
func (api *SysDatasetApi) GetDatasetOptions(c *gin.Context) {
	// 调用服务获取数据集下拉选项
	options, err := api.datasetService.GetDatasetOptions()
	if err != nil {
		common.FailWithMessage("获取数据集下拉选项失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(options, "查询成功", c)
}

// GetDatasetForm 数据集表单数据
// @Summary 数据集表单数据
// @Description 获取数据集表单数据
// @Tags 数据集接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "数据集ID"
// @Success 200 {object} vo.Result{data=bo.DatasetFormBO}
// @Router /api/v1/dataset/{id}/form [get]
func (api *SysDatasetApi) GetDatasetForm(c *gin.Context) {
	// 获取路径参数
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		common.FailWithMessage("数据集ID格式不正确", c)
		return
	}

	// 调用服务获取数据集表单数据
	datasetFormBO, err := api.datasetService.GetDatasetForm(id)
	if err != nil {
		common.FailWithMessage("获取数据集表单数据失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(datasetFormBO, "查询成功", c)
}

// SaveDataset 新增数据集
// @Summary 新增数据集
// @Description 新增数据集
// @Tags 数据集接口
// @Accept application/json
// @Produce application/json
// @Param datasetForm body bo.DatasetFormBO true "数据集表单数据"
// @Success 200 {object} vo.Result
// @Router /api/v1/dataset [post]
func (api *SysDatasetApi) SaveDataset(c *gin.Context) {
	// 绑定请求参数
	var datasetFormBO bo.DatasetFormBO
	if err := c.ShouldBindJSON(&datasetFormBO); err != nil {
		common.FailWithMessage("请求参数解析失败: "+err.Error(), c)
		return
	}

	// 调用服务保存数据集
	err := api.datasetService.SaveDataset(datasetFormBO)
	if err != nil {
		common.FailWithMessage("新增数据集失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("新增数据集成功", c)
}

// UpdateDataset 修改数据集
// @Summary 修改数据集
// @Description 修改数据集
// @Tags 数据集接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "数据集ID"
// @Param datasetForm body bo.DatasetFormBO true "数据集表单数据"
// @Success 200 {object} vo.Result
// @Router /api/v1/dataset/{id} [put]
func (api *SysDatasetApi) UpdateDataset(c *gin.Context) {
	// 获取路径参数
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		common.FailWithMessage("数据集ID格式不正确", c)
		return
	}

	// 绑定请求参数
	var datasetFormBO bo.DatasetFormBO
	if err := c.ShouldBindJSON(&datasetFormBO); err != nil {
		common.FailWithMessage("请求参数解析失败: "+err.Error(), c)
		return
	}

	// 调用服务更新数据集
	err = api.datasetService.UpdateDataset(id, datasetFormBO)
	if err != nil {
		common.FailWithMessage("修改数据集失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("修改数据集成功", c)
}

// DeleteDatasets 删除数据集
// @Summary 删除数据集
// @Description 删除数据集
// @Tags 数据集接口
// @Accept application/json
// @Produce application/json
// @Param ids query string true "数据集ID，多个以英文逗号(,)分割"
// @Success 200 {object} vo.Result
// @Router /api/v1/dataset [delete]
func (api *SysDatasetApi) DeleteDatasets(c *gin.Context) {
	// 获取查询参数
	idsStr := c.Query("ids")
	if idsStr == "" {
		common.FailWithMessage("数据集ID不能为空", c)
		return
	}

	// 解析ID列表
	idStrings := strings.Split(idsStr, ",")
	var ids []int64
	for _, idStr := range idStrings {
		id, err := strconv.ParseInt(idStr, 10, 64)
		if err != nil {
			common.FailWithMessage("数据集ID格式不正确", c)
			return
		}
		ids = append(ids, id)
	}

	// 调用服务删除数据集
	err := api.datasetService.DeleteDatasets(ids)
	if err != nil {
		common.FailWithMessage("删除数据集失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("删除数据集成功", c)
}

package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	datasetservice "github.com/earthyzinc/dehaze-go/internal/service/dataset"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/gin-gonic/gin"
)

type SysDatasetApi struct {
	datasetService   *datasetservice.DatasetService
	operationService *datasetservice.DatasetOperationService
}

func NewSysDatasetApi(datasetService *datasetservice.DatasetService, operationService *datasetservice.DatasetOperationService) *SysDatasetApi {
	return &SysDatasetApi{
		datasetService:   datasetService,
		operationService: operationService,
	}
}

// GetDatasetList 数据集分页列表
// @Summary 分页查询数据集列表
// @Tags 数据集接口
// @Accept application/json
// @Produce application/json
// @Param pageNum query int false "页码"
// @Param pageSize query int false "每页数量"
// @Param keyword query string false "关键字"
// @Param type query string false "类型"
// @Param status query int false "状态"
// @Success 200 {object} common.Response{data=vo.PageResult[vo.DatasetVO]}
// @Router /api/v1/datasets [get]
func (api *SysDatasetApi) GetDatasetList(c *gin.Context) {
	ctx := c.Request.Context()
	var queryParams query.DatasetQuery
	if err := c.ShouldBindQuery(&queryParams); err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "参数绑定失败"))
		return
	}

	result, err := api.datasetService.GetPage(ctx, &queryParams)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(result, "查询成功", c)
}

// GetDatasetChildren 获取子数据集列表（懒加载）
// @Summary 获取子数据集列表
// @Tags 数据集接口
// @Accept application/json
// @Produce application/json
// @Param parentId path int true "父数据集ID"
// @Success 200 {object} common.Response{data=[]vo.DatasetVO}
// @Router /api/v1/datasets/{parentId}/children [get]
func (api *SysDatasetApi) GetDatasetChildren(c *gin.Context) {
	ctx := c.Request.Context()
	parentIdStr := c.Param("parentId")
	parentId, err := strconv.ParseInt(parentIdStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "父数据集ID格式不正确"))
		return
	}

	children, err := api.datasetService.GetChildren(ctx, parentId)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(children, "查询成功", c)
}

// GetDatasetOptions 数据集下拉选项
// @Summary 获取数据集下拉选项
// @Tags 数据集接口
// @Accept application/json
// @Produce application/json
// @Success 200 {object} common.Response{data=[]vo.Option}
// @Router /api/v1/datasets/options [get]
func (api *SysDatasetApi) GetDatasetOptions(c *gin.Context) {
	options, err := api.datasetService.GetDatasetOptions()
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(options, "查询成功", c)
}

// GetDatasetById 获取数据集详情
// @Summary 获取数据集详情
// @Tags 数据集接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "数据集ID"
// @Success 200 {object} common.Response{data=bo.DatasetFormBO}
// @Router /api/v1/datasets/{id} [get]
func (api *SysDatasetApi) GetDatasetById(c *gin.Context) {
	ctx := c.Request.Context()
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "数据集ID格式不正确"))
		return
	}

	datasetFormBO, err := api.datasetService.GetFormData(ctx, id)
	if err != nil {
		_ = c.Error(err)
		return
	}

	if datasetFormBO.ID == nil {
		common.OkWithDetailed(nil, "查询成功", c)
		return
	}

	common.OkWithDetailed(datasetFormBO, "查询成功", c)
}

// SaveDataset 新增数据集
// @Summary 新增数据集
// @Tags 数据集接口
// @Accept application/json
// @Produce application/json
// @Param datasetForm body bo.DatasetFormBO true "数据集表单数据"
// @Success 200 {object} common.Response
// @Router /api/v1/datasets [post]
func (api *SysDatasetApi) SaveDataset(c *gin.Context) {
	ctx := c.Request.Context()
	var datasetFormBO bo.DatasetFormBO
	if err := c.ShouldBindJSON(&datasetFormBO); err != nil {
		_ = c.Error(err)
		return
	}

	if err := api.datasetService.Create(ctx, &datasetFormBO); err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithMessage("新增数据集成功", c)
}

// UpdateDataset 修改数据集
// @Summary 修改数据集
// @Tags 数据集接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "数据集ID"
// @Param datasetForm body bo.DatasetFormBO true "数据集表单数据"
// @Success 200 {object} common.Response
// @Router /api/v1/datasets/{id} [put]
func (api *SysDatasetApi) UpdateDataset(c *gin.Context) {
	ctx := c.Request.Context()
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "数据集ID格式不正确"))
		return
	}

	var datasetFormBO bo.DatasetFormBO
	if err := c.ShouldBindJSON(&datasetFormBO); err != nil {
		_ = c.Error(err)
		return
	}

	err = api.datasetService.Update(ctx, id, &datasetFormBO)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithMessage("修改数据集成功", c)
}

// DeleteDataset 删除单个数据集
// @Summary 删除单个数据集
// @Tags 数据集接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "数据集ID"
// @Success 200 {object} common.Response
// @Router /api/v1/datasets/{id} [delete]
func (api *SysDatasetApi) DeleteDataset(c *gin.Context) {
	ctx := c.Request.Context()
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "数据集ID格式不正确"))
		return
	}

	err = api.datasetService.Delete(ctx, []int64{id})
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithMessage("删除数据集成功", c)
}

// BatchDeleteDatasets 批量删除数据集
// @Summary 批量删除数据集
// @Tags 数据集接口
// @Accept application/json
// @Produce application/json
// @Param request body bo.BatchDeleteForm true "批量删除请求"
// @Success 200 {object} common.Response
// @Router /api/v1/datasets/batch [delete]
func (api *SysDatasetApi) BatchDeleteDatasets(c *gin.Context) {
	var req bo.BatchDeleteForm
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.operationService.BatchDeleteDatasets(c.Request.Context(), req)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithData(result, c)
}

package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	datasetservice "github.com/earthyzinc/dehaze-go/internal/service/dataset"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type SysDatasetItemApi struct {
	datasetItemService *datasetservice.DatasetItemService
	operationService   *datasetservice.DatasetOperationService
}

func NewSysDatasetItemApi(datasetItemService *datasetservice.DatasetItemService, operationService *datasetservice.DatasetOperationService) *SysDatasetItemApi {
	return &SysDatasetItemApi{
		datasetItemService: datasetItemService,
		operationService:   operationService,
	}
}

// GetDatasetItemById 获取数据项详情
// @Summary 获取数据项详情
// @Tags 数据项接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "数据项ID"
// @Success 200 {object} common.Response
// @Router /api/v1/dataset-items/{id} [get]
func (api *SysDatasetItemApi) GetDatasetItemById(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "数据项ID格式不正确"))
		return
	}

	itemVO, err := api.datasetItemService.GetDatasetItemVOByID(id)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(itemVO, "查询成功", c)
}

// GetDatasetItems 分页查询数据项列表
// @Summary 分页查询数据项列表
// @Tags 数据项接口
// @Accept application/json
// @Produce application/json
// @Param pageNum query int false "页码" default(1)
// @Param pageSize query int false "每页数量" default(10)
// @Param datasetId query int false "数据集ID"
// @Param sceneType query string false "场景类型"
// @Param keyword query string false "关键字（搜索文件名/描述）"
// @Param hazeLevel query string false "雾霾程度（light/medium/heavy）"
// @Success 200 {object} common.Response{data=[]vo.ImageItemVO}
// @Router /api/v1/dataset-items [get]
func (api *SysDatasetItemApi) GetDatasetItems(c *gin.Context) {
	pageNumStr := c.DefaultQuery("pageNum", "1")
	pageSizeStr := c.DefaultQuery("pageSize", "10")
	pageNum, _ := strconv.ParseInt(pageNumStr, 10, 64)
	pageSize, _ := strconv.ParseInt(pageSizeStr, 10, 64)

	datasetIdStr := c.Query("datasetId")
	sceneType := c.Query("sceneType")
	keyword := c.Query("keyword")
	hazeLevel := c.Query("hazeLevel")

	var datasetId int64
	var err error
	if datasetIdStr != "" {
		datasetId, err = strconv.ParseInt(datasetIdStr, 10, 64)
		if err != nil {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "数据集ID格式不正确"))
			return
		}
	}

	items, total, err := api.datasetItemService.GetDatasetItemsByPage(int(pageNum), int(pageSize), datasetId, sceneType, keyword, hazeLevel)
	if err != nil {
		_ = c.Error(err)
		return
	}

	result := map[string]interface{}{
		"records": items,
		"total":   total,
		"current": pageNum,
		"size":    pageSize,
	}

	common.OkWithDetailed(result, "查询成功", c)
}

// createDatasetItemRequest 创建空数据项请求
type createDatasetItemRequest struct {
	DatasetID int64  `json:"datasetId" binding:"required"`
	Name      string `json:"name"`
}

// CreateDatasetItem 创建空数据项
// @Summary 创建空数据项
// @Tags 数据项接口
// @Accept application/json
// @Produce application/json
// @Param request body createDatasetItemRequest true "创建请求"
// @Success 200 {object} common.Response{data=int64}
// @Router /api/v1/dataset-items [post]
func (api *SysDatasetItemApi) CreateDatasetItem(c *gin.Context) {
	var req createDatasetItemRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	var datasetItem interface{}
	var err error

	if req.Name != "" {
		item, e := api.datasetItemService.CreateDatasetItemWithName(req.DatasetID, req.Name)
		datasetItem = item.ID
		err = e
	} else {
		item, e := api.datasetItemService.CreateDatasetItem(req.DatasetID)
		datasetItem = item.ID
		err = e
	}

	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithData(datasetItem, c)
}

// uploadDatasetItemRequest 创建数据项并上传配对图片的请求（转发到 operationService）
// 复用 datasetservice.CreateDatasetItemWithImagesRequest

// CreateDatasetItemWithImages 创建数据项并上传配对图片
// @Summary 创建数据项并上传配对图片
// @Tags 数据项接口
// @Accept application/json
// @Produce application/json
// @Param request body datasetservice.CreateDatasetItemWithImagesRequest true "创建请求"
// @Success 200 {object} common.Response
// @Router /api/v1/dataset-items/upload [post]
func (api *SysDatasetItemApi) CreateDatasetItemWithImages(c *gin.Context) {
	var req datasetservice.CreateDatasetItemWithImagesRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.operationService.CreateDatasetItemWithImages(c.Request.Context(), req)
	if err != nil {
		logger.Error("创建数据项失败", zap.Error(err))
		_ = c.Error(err)
		return
	}

	common.OkWithData(result, c)
}

// BatchCreateDatasetItemsWithImages 批量创建数据项并上传图片
// @Summary 批量创建数据项并上传图片
// @Tags 数据项接口
// @Accept application/json
// @Produce application/json
// @Param request body datasetservice.BatchCreateDatasetItemsWithImagesRequest true "批量创建请求"
// @Success 200 {object} common.Response
// @Router /api/v1/dataset-items/batch [post]
func (api *SysDatasetItemApi) BatchCreateDatasetItemsWithImages(c *gin.Context) {
	var req datasetservice.BatchCreateDatasetItemsWithImagesRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	result, err := api.operationService.BatchCreateDatasetItemsWithImages(c.Request.Context(), req)
	if err != nil {
		logger.Error("批量创建数据项失败", zap.Error(err))
		_ = c.Error(err)
		return
	}

	common.OkWithData(result, c)
}

// updateDatasetItemRequest 修改数据项请求
type updateDatasetItemRequest struct {
	Name string `json:"name"`
}

// UpdateDatasetItem 修改数据项
// @Summary 修改数据项
// @Tags 数据项接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "数据项ID"
// @Param request body updateDatasetItemRequest true "修改请求"
// @Success 200 {object} common.Response
// @Router /api/v1/dataset-items/{id} [put]
func (api *SysDatasetItemApi) UpdateDatasetItem(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "数据项ID格式不正确"))
		return
	}

	var req updateDatasetItemRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	err = api.datasetItemService.UpdateDatasetItem(id, req.Name)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithMessage("修改数据项成功", c)
}

// DeleteDatasetItem 删除数据项（级联删除关联文件）
// @Summary 删除数据项
// @Tags 数据项接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "数据项ID"
// @Success 200 {object} common.Response
// @Router /api/v1/dataset-items/{id} [delete]
func (api *SysDatasetItemApi) DeleteDatasetItem(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "数据项ID格式不正确"))
		return
	}

	err = api.operationService.DeleteDatasetItemCascade(c.Request.Context(), id)
	if err != nil {
		logger.Error("删除数据项失败", zap.Error(err))
		_ = c.Error(err)
		return
	}

	common.OkWithMessage("删除成功", c)
}

// BatchDeleteDatasetItems 批量删除数据项
// @Summary 批量删除数据项
// @Tags 数据项接口
// @Accept application/json
// @Produce application/json
// @Param request body bo.BatchDeleteForm true "批量删除请求"
// @Success 200 {object} common.Response
// @Router /api/v1/dataset-items/batch [delete]
func (api *SysDatasetItemApi) BatchDeleteDatasetItems(c *gin.Context) {
	var req bo.BatchDeleteForm
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	for _, id := range req.IDs {
		if err := api.operationService.DeleteDatasetItemCascade(c.Request.Context(), id); err != nil {
			logger.Error("批量删除数据项失败", zap.Int64("itemID", id), zap.Error(err))
			_ = c.Error(err)
			return
		}
	}

	common.OkWithMessage("批量删除成功", c)
}

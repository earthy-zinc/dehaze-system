package api

import (
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/service"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// DatasetOperationApi 数据集操作API
type DatasetOperationApi struct {
	operationService *service.DatasetOperationService
}

// NewDatasetOperationApi 创建数据集操作API实例
func NewDatasetOperationApi() *DatasetOperationApi {
	return &DatasetOperationApi{
		operationService: &service.DatasetOperationService{},
	}
}

// CreateDatasetItemWithImages 创建数据项并上传配对图片
// @Summary 创建数据项并上传配对图片
// @Tags 数据集操作
// @Accept json
// @Produce json
// @Param request body service.CreateDatasetItemWithImagesRequest true "创建请求"
// @Success 200 {object} common.Response{data=vo.ImageItemVO}
// @Router /api/v1/dataset/operations/items [post]
func (d *DatasetOperationApi) CreateDatasetItemWithImages(c *gin.Context) {
	var req service.CreateDatasetItemWithImagesRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	result, err := d.operationService.CreateDatasetItemWithImages(c.Request.Context(), req)
	if err != nil {
		logger.Error("创建数据项失败", zap.Error(err))
		common.FailWithMessage(err.Error(), c)
		return
	}

	common.OkWithData(result, c)
}

// BatchCreateDatasetItemsWithImages 批量创建数据项并上传配对图片
// @Summary 批量创建数据项并上传配对图片
// @Tags 数据集操作
// @Accept json
// @Produce json
// @Param request body service.BatchCreateDatasetItemsWithImagesRequest true "批量创建请求"
// @Success 200 {object} common.Response{data=service.BatchCreateResult}
// @Router /api/v1/dataset/operations/items/batch [post]
func (d *DatasetOperationApi) BatchCreateDatasetItemsWithImages(c *gin.Context) {
	var req service.BatchCreateDatasetItemsWithImagesRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	result, err := d.operationService.BatchCreateDatasetItemsWithImages(c.Request.Context(), req)
	if err != nil {
		logger.Error("批量创建数据项失败", zap.Error(err))
		common.FailWithMessage(err.Error(), c)
		return
	}

	common.OkWithData(result, c)
}

// DeleteDatasetItemCascade 级联删除数据项
// @Summary 级联删除数据项
// @Tags 数据集操作
// @Accept json
// @Produce json
// @Param itemId path int true "数据项ID"
// @Success 200 {object} common.Response
// @Router /api/v1/dataset/operations/items/{itemId} [delete]
func (d *DatasetOperationApi) DeleteDatasetItemCascade(c *gin.Context) {
	var itemId bo.Id
	if err := c.ShouldBindUri(&itemId); err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	err := d.operationService.DeleteDatasetItemCascade(c.Request.Context(), itemId.ID)
	if err != nil {
		logger.Error("级联删除数据项失败", zap.Error(err))
		common.FailWithMessage(err.Error(), c)
		return
	}

	common.OkWithMessage("删除成功", c)
}

// BatchDeleteDatasets 批量删除数据集
// @Summary 批量删除数据集
// @Tags 数据集操作
// @Accept json
// @Produce json
// @Param request body bo.BatchDeleteForm true "批量删除请求"
// @Success 200 {object} common.Response{data=service.BatchCreateResult}
// @Router /api/v1/dataset/operations/batch [post]
func (d *DatasetOperationApi) BatchDeleteDatasets(c *gin.Context) {
	var req bo.BatchDeleteForm
	if err := c.ShouldBindJSON(&req); err != nil {
		common.FailWithMessage(err.Error(), c)
		return
	}

	result, err := d.operationService.BatchDeleteDatasets(c.Request.Context(), req)
	if err != nil {
		logger.Error("批量删除数据集失败", zap.Error(err))
		common.FailWithMessage(err.Error(), c)
		return
	}

	common.OkWithData(result, c)
}

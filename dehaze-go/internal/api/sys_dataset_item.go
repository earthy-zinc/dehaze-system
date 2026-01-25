package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/service"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/gin-gonic/gin"
)

type SysDatasetItemApi struct {
	datasetItemService service.DatasetItemService
}

// GetDatasetItemById 获取数据项详情
// @Summary 获取数据项详情
// @Description 根据ID获取数据项详情
// @Tags 数据集项接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "数据项ID"
// @Success 200 {object} common.Result
// @Router /api/v1/dataset-items/{id} [get]
func (api *SysDatasetItemApi) GetDatasetItemById(c *gin.Context) {
	// 获取路径参数
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		common.FailWithMessage("数据项ID格式不正确", c)
		return
	}

	// 调用服务获取数据项详情
	itemVO, err := api.datasetItemService.GetDatasetItemVOByID(id)
	if err != nil {
		common.FailWithMessage("获取数据项详情失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(itemVO, "查询成功", c)
}

// GetDatasetItems 分页查询数据项列表
// @Summary 分页查询数据项列表
// @Description 分页查询数据项列表
// @Tags 数据集项接口
// @Accept application/json
// @Produce application/json
// @Param pageNum query int false "页码" default(1)
// @Param pageSize query int false "每页数量" default(10)
// @Param datasetId query int false "数据集ID"
// @Param sceneType query string false "场景类型"
// @Success 200 {object} common.Result{data=[]vo.ImageItemVO}
// @Router /api/v1/dataset-items [get]
func (api *SysDatasetItemApi) GetDatasetItems(c *gin.Context) {
	// 获取分页参数
	pageNumStr := c.DefaultQuery("pageNum", "1")
	pageSizeStr := c.DefaultQuery("pageSize", "10")
	pageNum, _ := strconv.ParseInt(pageNumStr, 10, 64)
	pageSize, _ := strconv.ParseInt(pageSizeStr, 10, 64)

	// 获取筛选参数
	datasetIdStr := c.Query("datasetId")
	sceneType := c.Query("sceneType")

	var datasetId int64
	var err error
	if datasetIdStr != "" {
		datasetId, err = strconv.ParseInt(datasetIdStr, 10, 64)
		if err != nil {
			common.FailWithMessage("数据集ID格式不正确", c)
			return
		}
	}

	// 调用服务获取数据项列表
	items, total, err := api.datasetItemService.GetDatasetItemsByPage(int(pageNum), int(pageSize), datasetId, sceneType)
	if err != nil {
		common.FailWithMessage("获取数据项列表失败: "+err.Error(), c)
		return
	}

	// 构造分页响应
	result := map[string]interface{}{
		"records": items,
		"total":   total,
		"current": pageNum,
		"size":    pageSize,
	}

	common.OkWithDetailed(result, "查询成功", c)
}

// CreateDatasetItem 新增数据项
// @Summary 新增数据项
// @Description 新增数据项
// @Tags 数据集项接口
// @Accept application/json
// @Produce application/json
// @Param datasetId query int true "所属数据集ID"
// @Param name query string false "名称"
// @Success 200 {object} common.Result{data=int64}
// @Router /api/v1/dataset/item [post]
func (api *SysDatasetItemApi) CreateDatasetItem(c *gin.Context) {
	// 获取参数
	datasetIdStr := c.Query("datasetId")
	datasetId, err := strconv.ParseInt(datasetIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("数据集ID格式不正确", c)
		return
	}

	name := c.Query("name")

	// 调用服务创建数据项
	datasetItem, err := api.datasetItemService.CreateDatasetItemWithName(datasetId, name)
	if name != "" {
		datasetItem, err = api.datasetItemService.CreateDatasetItemWithName(datasetId, name)
	} else {
		datasetItem, err = api.datasetItemService.CreateDatasetItem(datasetId)
	}

	if err != nil {
		common.FailWithMessage("创建数据项失败: "+err.Error(), c)
		return
	}

	common.OkWithData(datasetItem.ID, c)
}

// UpdateDatasetItem 修改数据项
// @Summary 修改数据项
// @Description 修改数据项
// @Tags 数据集项接口
// @Accept application/json
// @Produce application/json
// @Param datasetItemId query int true "数据项ID"
// @Param name query string false "名称"
// @Success 200 {object} common.Result
// @Router /api/v1/dataset/item [put]
func (api *SysDatasetItemApi) UpdateDatasetItem(c *gin.Context) {
	// 获取参数
	datasetItemIdStr := c.Query("datasetItemId")
	datasetItemId, err := strconv.ParseInt(datasetItemIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("数据项ID格式不正确", c)
		return
	}

	name := c.Query("name")

	// 调用服务更新数据项
	err = api.datasetItemService.UpdateDatasetItem(datasetItemId, name)
	if err != nil {
		common.FailWithMessage("修改数据项失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("修改数据项成功", c)
}

// DeleteDatasetItem 删除数据项
// @Summary 删除数据项
// @Description 删除数据项
// @Tags 数据集项接口
// @Accept application/json
// @Produce application/json
// @Param datasetItemId query int true "数据项ID"
// @Success 200 {object} common.Result
// @Router /api/v1/dataset/item [delete]
func (api *SysDatasetItemApi) DeleteDatasetItem(c *gin.Context) {
	// 获取参数
	datasetItemIdStr := c.Query("datasetItemId")
	datasetItemId, err := strconv.ParseInt(datasetItemIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("数据项ID格式不正确", c)
		return
	}

	// 调用服务删除数据项
	err = api.datasetItemService.DeleteDatasetItem(datasetItemId)
	if err != nil {
		common.FailWithMessage("删除数据项失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("删除数据项成功", c)
}

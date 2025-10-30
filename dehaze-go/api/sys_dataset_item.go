package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/common"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/gin-gonic/gin"
)

type SysDatasetItemApi struct {
	datasetItemService service.DatasetItemService
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

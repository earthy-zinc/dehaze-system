package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/common"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/gin-gonic/gin"
)

type SysItemFileApi struct {
	itemFileService service.ItemFileService
}

// AddImageById 上传数据项图片
// @Summary 上传数据项图片
// @Description 上传数据项图片
// @Tags 数据集项接口
// @Accept multipart/form-data
// @Produce application/json
// @Param file formData file true "表单文件对象"
// @Param datasetId formData int true "所属数据集ID"
// @Param datasetItemId formData int true "所属数据项ID"
// @Param type formData string true "图片类型"
// @Param description formData string false "图片描述"
// @Success 200 {object} common.Result{data=dto.ImageFileInfo}
// @Router /api/v1/dataset/image [post]
func (api *SysItemFileApi) AddImageById(c *gin.Context) {
	// 获取上传的文件
	file, err := c.FormFile("file")
	if err != nil {
		common.FailWithMessage("文件上传失败: "+err.Error(), c)
		return
	}

	// 获取参数
	datasetItemIdStr := c.PostForm("datasetItemId")
	datasetItemId, err := strconv.ParseInt(datasetItemIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("数据项ID格式不正确", c)
		return
	}

	fileType := c.PostForm("type")
	description := c.PostForm("description")

	// 构建业务对象
	itemBO := bo.DatasetItemBO{
		FileBO: bo.FileBO{
			Name: file.Filename,
			// TODO: 需要根据实际文件处理逻辑填充其他字段
		},
		Type:        fileType,
		Description: description,
	}

	// 调用服务保存项文件
	imageFileInfo, err := api.itemFileService.SaveItemFile(datasetItemId, itemBO)
	if err != nil {
		common.FailWithMessage("保存项文件失败: "+err.Error(), c)
		return
	}

	common.OkWithData(imageFileInfo, c)
}

// UpdateImageById 修改数据项图片信息
// @Summary 修改数据项图片信息
// @Description 修改数据项图片信息
// @Tags 数据集项接口
// @Accept application/json
// @Produce application/json
// @Param itemFileId query int true "数据项文件ID"
// @Param type query string true "图片类型"
// @Param description query string false "图片描述"
// @Success 200 {object} common.Result{data=dto.ImageFileInfo}
// @Router /api/v1/dataset/image [put]
func (api *SysItemFileApi) UpdateImageById(c *gin.Context) {
	// 获取参数
	itemFileIdStr := c.Query("itemFileId")
	_, err := strconv.ParseInt(itemFileIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("数据项文件ID格式不正确", c)
		return
	}

	_ = c.Query("type")
	_ = c.Query("description")

	// TODO: 实现更新逻辑
	common.FailWithMessage("暂未实现", c)
}

// RemoveImageById 删除数据项图片
// @Summary 删除数据项图片
// @Description 删除数据项图片
// @Tags 数据集项接口
// @Accept application/json
// @Produce application/json
// @Param itemFileId query int true "数据项文件ID"
// @Success 200 {object} common.Result
// @Router /api/v1/dataset/image [delete]
func (api *SysItemFileApi) RemoveImageById(c *gin.Context) {
	// 获取参数
	itemFileIdStr := c.Query("itemFileId")
	itemFileId, err := strconv.ParseInt(itemFileIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("数据项文件ID格式不正确", c)
		return
	}

	// 调用服务删除项文件
	err = api.itemFileService.DeleteItemFile(itemFileId)
	if err != nil {
		common.FailWithMessage("删除数据项图片失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("删除数据项图片成功", c)
}
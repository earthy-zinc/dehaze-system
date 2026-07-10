package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	fileservice "github.com/earthyzinc/dehaze-go/internal/service/file"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/gin-gonic/gin"
)

type SysItemFileApi struct {
	itemFileService *fileservice.ItemFileService
}

func NewSysItemFileApi(itemFileService *fileservice.ItemFileService) *SysItemFileApi {
	return &SysItemFileApi{
		itemFileService: itemFileService,
	}
}

// GetItemFileById 获取图片详细信息
// @Summary 获取图片详细信息
// @Tags 图片文件接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "图片文件ID"
// @Success 200 {object} common.Response
// @Router /api/v1/item-files/{id} [get]
func (api *SysItemFileApi) GetItemFileById(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "图片文件ID格式不正确"))
		return
	}

	itemFile, err := api.itemFileService.GetItemFileById(id)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(itemFile, "查询成功", c)
}

// AddImageById 上传数据项图片
// @Summary 上传数据项图片
// @Tags 图片文件接口
// @Accept multipart/form-data
// @Produce application/json
// @Param file formData file true "表单文件对象"
// @Param datasetItemId formData int true "所属数据项ID"
// @Param type formData string true "图片类型"
// @Param description formData string false "图片描述"
// @Success 200 {object} common.Response{data=dto.ImageFileInfo}
// @Router /api/v1/item-files [post]
func (api *SysItemFileApi) AddImageById(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "文件上传失败"))
		return
	}

	datasetItemIdStr := c.PostForm("datasetItemId")
	datasetItemId, err := strconv.ParseInt(datasetItemIdStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "数据项ID格式不正确"))
		return
	}

	fileType := c.PostForm("type")
	description := c.PostForm("description")

	itemBO := bo.DatasetItemBO{
		FileBO: bo.FileBO{
			Name: file.Filename,
		},
		Type:        fileType,
		Description: description,
	}

	imageFileInfo, err := api.itemFileService.SaveItemFile(datasetItemId, itemBO, true)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithData(imageFileInfo, c)
}

// UpdateImageById 修改图片信息
// @Summary 修改图片信息
// @Tags 图片文件接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "图片文件ID"
// @Success 200 {object} common.Response
// @Router /api/v1/item-files/{id} [put]
func (api *SysItemFileApi) UpdateImageById(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "图片文件ID格式不正确"))
		return
	}

	var form bo.ItemFileUpdateForm
	if err := c.ShouldBindJSON(&form); err != nil {
		_ = c.Error(err)
		return
	}

	err = api.itemFileService.UpdateItemFileInfo(id, form)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithMessage("更新成功", c)
}

// RemoveImageById 删除图片
// @Summary 删除图片
// @Tags 图片文件接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "图片文件ID"
// @Success 200 {object} common.Response
// @Router /api/v1/item-files/{id} [delete]
func (api *SysItemFileApi) RemoveImageById(c *gin.Context) {
	idStr := c.Param("id")
	itemFileId, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "图片文件ID格式不正确"))
		return
	}

	err = api.itemFileService.DeleteItemFile(itemFileId)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithMessage("删除图片成功", c)
}

// BatchRemoveImages 批量删除图片
// @Summary 批量删除图片
// @Tags 图片文件接口
// @Accept application/json
// @Produce application/json
// @Param request body bo.BatchDeleteForm true "批量删除请求"
// @Success 200 {object} common.Response
// @Router /api/v1/item-files/batch [delete]
func (api *SysItemFileApi) BatchRemoveImages(c *gin.Context) {
	var req bo.BatchDeleteForm
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	for _, id := range req.IDs {
		if err := api.itemFileService.DeleteItemFile(id); err != nil {
			_ = c.Error(err)
			return
		}
	}

	common.OkWithMessage("批量删除图片成功", c)
}

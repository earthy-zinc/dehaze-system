package api

import (
	"fmt"
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	fileservice "github.com/earthyzinc/dehaze-go/internal/service/file"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/gin-gonic/gin"
)

type SysItemFileApi struct {
	itemFileService *fileservice.ItemFileService
	fileService     *fileservice.FileService
}

func NewSysItemFileApi(itemFileService *fileservice.ItemFileService, fileService *fileservice.FileService) *SysItemFileApi {
	return &SysItemFileApi{
		itemFileService: itemFileService,
		fileService:     fileService,
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
// @Param itemId formData int true "所属数据项ID"
// @Param type formData string true "图片类型"
// @Param description formData string false "图片描述"
// @Success 200 {object} common.Response{data=vo.ImageUrlVO}
// @Router /api/v1/item-files [post]
func (api *SysItemFileApi) AddImageById(c *gin.Context) {
	ctx := c.Request.Context()

	fileHeader, err := c.FormFile("file")
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "文件上传失败"))
		return
	}

	itemIdStr := c.PostForm("itemId")
	datasetItemId, err := strconv.ParseInt(itemIdStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "数据项ID格式不正确"))
		return
	}

	fileType := c.PostForm("type")
	description := c.PostForm("description")
	sceneType := c.PostForm("sceneType")
	hazeLevel := c.PostForm("hazeLevel")

	// 打开文件流并计算 MD5
	file, err := fileHeader.Open()
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "无法读取文件"))
		return
	}
	defer file.Close()

	md5Hash, reader, err := fileservice.ComputeMD5(file)
	if err != nil {
		_ = c.Error(common.WrapBizError(common.SYSTEM_RESOURCE_ACCESS_ERR, "计算文件MD5失败", err))
		return
	}

	// 上传文件
	baseURL := fmt.Sprintf("http://%s/api/v1/files/download", c.Request.Host)
	sysFile, err := api.fileService.UploadFile(ctx, fileHeader, reader, md5Hash, baseURL)
	if err != nil {
		_ = c.Error(err)
		return
	}

	// 构建业务对象并保存项文件关联
	itemBO := bo.DatasetItemBO{
		FileBO: bo.FileBO{
			Name: fileHeader.Filename,
		},
		Type:        fileType,
		Description: description,
		SceneType:   sceneType,
		HazeLevel:   hazeLevel,
	}

	imageUrlVO, err := api.itemFileService.SaveItemFile(datasetItemId, sysFile, itemBO, true)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(imageUrlVO, "上传成功", c)
}

// UpdateImageById 修改图片信息
// @Summary 修改图片信息
// @Tags 图片文件接口
// @Accept application/json
// @Produce application/json
// @Param id path int true "图片文件ID"
// @Success 200 {object} common.Response{data=vo.ImageUrlVO}
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

	imageUrlVO, err := api.itemFileService.UpdateItemFileInfo(id, form)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(imageUrlVO, "更新成功", c)
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
// @Success 200 {object} common.Response{data=batchImageDeleteResult}
// @Router /api/v1/item-files/batch [delete]
func (api *SysItemFileApi) BatchRemoveImages(c *gin.Context) {
	var req bo.BatchDeleteForm
	if err := c.ShouldBindJSON(&req); err != nil {
		_ = c.Error(err)
		return
	}

	successIds := make([]int64, 0, len(req.IDs))
	failedItems := make([]batchImageDeleteFailure, 0)
	successCount := 0
	failedCount := 0

	for _, id := range req.IDs {
		if err := api.itemFileService.DeleteItemFile(id); err != nil {
			failedCount++
			failedItems = append(failedItems, batchImageDeleteFailure{
				ID:     id,
				Reason: err.Error(),
			})
			continue
		}
		successCount++
		successIds = append(successIds, id)
	}

	result := batchImageDeleteResult{
		SuccessCount: successCount,
		FailedCount:  failedCount,
		SuccessIds:   successIds,
		FailedItems:  failedItems,
	}

	common.OkWithDetailed(result, "批量删除完成", c)
}

// batchImageDeleteResult 批量删除图片结果
type batchImageDeleteResult struct {
	SuccessCount int                       `json:"successCount"`
	FailedCount  int                       `json:"failedCount"`
	SuccessIds   []int64                   `json:"successIds,omitempty"`
	FailedItems  []batchImageDeleteFailure `json:"failedItems,omitempty"`
}

// batchImageDeleteFailure 批量删除失败项
type batchImageDeleteFailure struct {
	ID     int64  `json:"id"`
	Reason string `json:"reason"`
}

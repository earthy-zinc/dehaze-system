package api

import (
	"fmt"
	"path/filepath"
	"strconv"
	"time"

	"github.com/earthyzinc/dehaze-go/common"
	"github.com/earthyzinc/dehaze-go/service"
	"github.com/gin-gonic/gin"
)

type SysFileApi struct {
	sysFileService service.SysFileService
	fileService    service.FileService
}

// UploadFile 文件上传
// @Summary 文件上传
// @Description 文件上传
// @Tags 文件接口
// @Accept multipart/form-data
// @Produce application/json
// @Param file formData file true "表单文件对象"
// @Param modelId formData int false "模型id"
// @Success 200 {object} common.Result{data=model.SysFile}
// @Router /api/v1/files [post]
func (api *SysFileApi) UploadFile(c *gin.Context) {
	// 获取上传的文件
	file, err := c.FormFile("file")
	if err != nil {
		common.FailWithMessage("文件上传失败: "+err.Error(), c)
		return
	}

	// 获取模型ID参数
	modelIdStr := c.PostForm("modelId")
	var modelId *int64
	if modelIdStr != "" {
		id, err := strconv.ParseInt(modelIdStr, 10, 64)
		if err != nil {
			common.FailWithMessage("模型ID格式不正确", c)
			return
		}
		modelId = &id
	}

	// 构建上传路径
	uploadPath := "upload/" + time.Now().Format("20060102")
	baseUrl := c.Request.Host

	// 上传文件
	fileBO, err := api.fileService.UploadFile(file, baseUrl, uploadPath)
	if err != nil {
		common.FailWithMessage("文件上传失败: "+err.Error(), c)
		return
	}

	// 保存文件信息到数据库
	sysFile, err := api.sysFileService.SaveFile(fileBO)
	if err != nil {
		common.FailWithMessage("保存文件信息失败: "+err.Error(), c)
		return
	}

	// 如果有模型ID，则获取WPX文件
	if modelId != nil {
		// TODO: 实现获取WPX文件的逻辑
		// 这里暂时直接返回原始文件
		common.OkWithData(sysFile, c)
		return
	}

	common.OkWithData(sysFile, c)
}

// DeleteFile 文件删除
// @Summary 文件删除
// @Description 文件删除
// @Tags 文件接口
// @Accept application/json
// @Produce application/json
// @Param fileId query int true "文件ID"
// @Success 200 {object} common.Result
// @Router /api/v1/files [delete]
func (api *SysFileApi) DeleteFile(c *gin.Context) {
	// 获取文件ID参数
	fileIdStr := c.Query("fileId")
	fileId, err := strconv.ParseInt(fileIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("文件ID格式不正确", c)
		return
	}

	// 删除文件
	err = api.sysFileService.DeleteFile(fileId)
	if err != nil {
		common.FailWithMessage("删除文件失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("删除成功", c)
}

// CheckFile 文件校验
// @Summary 文件校验
// @Description 文件校验
// @Tags 文件接口
// @Accept application/json
// @Produce application/json
// @Param md5 query string true "文件md5"
// @Success 200 {object} common.Result{data=bool}
// @Router /api/v1/files/check [get]
func (api *SysFileApi) CheckFile(c *gin.Context) {
	// 获取MD5参数
	md5 := c.Query("md5")
	if md5 == "" {
		common.FailWithMessage("缺少md5参数", c)
		return
	}

	// 校验文件
	result := api.sysFileService.CheckFile(md5)
	common.OkWithData(result, c)
}

// DownloadFile 文件下载
// @Summary 文件下载
// @Description 文件下载
// @Tags 文件接口
// @Accept application/json
// @Produce application/json
// @Param objectName path string true "对象存储名称"
// @Success 200 {object} common.Result
// @Router /api/v1/files/download/{objectName} [get]
func (api *SysFileApi) DownloadFile(c *gin.Context) {
	// 获取对象存储名称
	objectName := c.Param("objectName")
	if objectName == "" {
		common.FailWithMessage("缺少objectName参数", c)
		return
	}

	// 获取文件路径
	filePath, err := api.sysFileService.DownloadFile(objectName)
	if err != nil {
		common.FailWithMessage("下载文件失败: "+err.Error(), c)
		return
	}

	// 提取文件名
	filename := filepath.Base(objectName)

	// 设置响应头
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", filename))
	c.Header("Content-Type", "application/octet-stream")

	// 返回文件
	c.File(filePath)
}

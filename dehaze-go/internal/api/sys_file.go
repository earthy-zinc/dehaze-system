package api

import (
	"fmt"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"

	fileservice "github.com/earthyzinc/dehaze-go/internal/service/file"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/gin-gonic/gin"
)

const defaultMaxFileSize = int64(100 * 1024 * 1024) // 100MB

type SysFileApi struct {
	fileService *fileservice.FileService
}

func NewSysFileApi(fileService *fileservice.FileService) *SysFileApi {
	return &SysFileApi{
		fileService: fileService,
	}
}

// UploadFile 文件上传
// @Summary 文件上传
// @Description 文件上传（支持秒传：MD5 命中则直接返回已有记录）
// @Tags 文件接口
// @Accept multipart/form-data
// @Produce application/json
// @Param file formData file true "表单文件对象"
// @Success 200 {object} common.Response{data=model.SysFile}
// @Router /api/v1/files [post]
func (api *SysFileApi) UploadFile(c *gin.Context) {
	ctx := c.Request.Context()

	// 1. 获取上传的文件
	fileHeader, err := c.FormFile("file")
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "文件上传失败"))
		return
	}

	// 2. 文件大小校验
	maxSize := defaultMaxFileSize
	if cfg := config.GetConfig(); cfg != nil && cfg.File.MaxSize > 0 {
		maxSize = cfg.File.MaxSize
	}
	if fileHeader.Size > maxSize {
		_ = c.Error(common.NewBizError(common.USER_UPLOAD_FILE_SIZE_EXCEEDS, "文件大小超过限制"))
		return
	}

	// 3. 文件名安全校验
	if err := validateFileName(fileHeader.Filename); err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, err.Error()))
		return
	}

	// 4. 打开文件流并计算 MD5
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

	// 5. 构建 baseURL
	baseURL := fmt.Sprintf("http://%s/api/v1/files/download", c.Request.Host)

	// 6. 调用 Service 上传
	sysFile, err := api.fileService.UploadFile(ctx, fileHeader, reader, md5Hash, baseURL)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response
// @Router /api/v1/files [delete]
func (api *SysFileApi) DeleteFile(c *gin.Context) {
	fileIdStr := c.Query("fileId")
	fileId, err := strconv.ParseInt(fileIdStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "文件ID格式不正确"))
		return
	}

	err = api.fileService.DeleteFile(c.Request.Context(), fileId)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithMessage("删除成功", c)
}

// CheckFile 文件校验
// @Summary 文件校验
// @Description 根据 MD5 校验文件是否已存在（用于秒传预检）
// @Tags 文件接口
// @Accept application/json
// @Produce application/json
// @Param md5 query string true "文件md5"
// @Success 200 {object} common.Response{data=bool}
// @Router /api/v1/files/check [get]
func (api *SysFileApi) CheckFile(c *gin.Context) {
	md5 := c.Query("md5")
	if md5 == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "缺少md5参数"))
		return
	}

	result, err := api.fileService.CheckFile(c.Request.Context(), md5)
	if err != nil {
		_ = c.Error(err)
		return
	}
	// result 为 *model.SysFile 类型的 nil 指针时，传入 OkWithData 会因 typed-nil 生成 "data":null，
	// 这里显式判断，文件不存在时返回不带 data 字段的成功响应，SDK 侧得到 undefined
	if result == nil {
		common.Ok(c)
		return
	}
	common.OkWithData(result, c)
}

// GetFilePage 分页查询文件列表
// @Summary 分页查询文件列表
// @Description 分页查询文件列表
// @Tags 文件接口
// @Accept application/json
// @Produce application/json
// @Param pageNum query int false "页码" default(1)
// @Param pageSize query int false "每页数量" default(10)
// @Param keywords query string false "关键字(文件名/类型)"
// @Success 200 {object} common.Response{data=common.PageResult}
// @Router /api/v1/files/page [get]
func (api *SysFileApi) GetFilePage(c *gin.Context) {
	ctx := c.Request.Context()

	pageNum := 1
	pageSize := 10
	if v := c.Query("pageNum"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			pageNum = n
		}
	}
	if v := c.Query("pageSize"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			pageSize = n
		}
	}
	keywords := c.Query("keywords")

	result, err := api.fileService.GetPage(ctx, pageNum, pageSize, keywords)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(result, common.SUCCESS.Msg, c)
}

// GetFileDetail 获取文件详情
// @Summary 获取文件详情
// @Description 获取文件详情
// @Tags 文件接口
// @Accept application/json
// @Produce application/json
// @Param fileId path int true "文件ID"
// @Success 200 {object} common.Response{data=model.SysFile}
// @Router /api/v1/files/{fileId} [get]
func (api *SysFileApi) GetFileDetail(c *gin.Context) {
	fileIdStr := c.Param("fileId")
	fileId, err := strconv.ParseInt(fileIdStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "文件ID格式不正确"))
		return
	}

	file, err := api.fileService.GetFileById(c.Request.Context(), fileId)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithData(file, c)
}

// DownloadFile 文件下载
// @Summary 文件下载
// @Description 文件下载（流式传输）
// @Tags 文件接口
// @Accept application/json
// @Produce application/octet-stream
// @Param objectName path string true "对象存储名称"
// @Success 200 {object} common.Response
// @Router /api/v1/files/download/{objectName} [get]
func (api *SysFileApi) DownloadFile(c *gin.Context) {
	ctx := c.Request.Context()

	objectName := c.Param("objectName")
	if objectName == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "缺少objectName参数"))
		return
	}

	file, err := api.fileService.GetFileByObjectName(ctx, objectName)
	if err == nil && file != nil && file.URL != nil && *file.URL != "" {
		if cfg := config.GetConfig(); cfg != nil && !strings.HasPrefix(*file.URL, cfg.File.BaseURL) {
			c.Redirect(http.StatusFound, *file.URL)
			return
		}
	}

	reader, sysFile, err := api.fileService.DownloadFile(ctx, objectName)
	if err != nil {
		_ = c.Error(err)
		return
	}
	defer reader.Close()

	filename := filepath.Base(objectName)
	if sysFile != nil && sysFile.Name != "" {
		filename = sysFile.Name
	}

	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", filename))
	c.Header("Content-Type", "application/octet-stream")
	c.DataFromReader(200, -1, "application/octet-stream", reader, nil)
}

// validateFileName 校验文件名安全性
func validateFileName(fileName string) error {
	if fileName == "" {
		return fmt.Errorf("文件名不能为空")
	}
	if len(fileName) > 255 {
		return fmt.Errorf("文件名过长")
	}
	// 禁止路径分隔符防止路径穿越
	for _, ch := range fileName {
		if ch == '/' || ch == '\\' || ch == '\x00' {
			return fmt.Errorf("文件名包含非法字符")
		}
	}
	return nil
}

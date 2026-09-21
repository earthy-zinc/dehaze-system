package api

import (
	"context"
	"fmt"
	"mime"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/earthyzinc/dehaze-go/internal/model"
	fileservice "github.com/earthyzinc/dehaze-go/internal/service/file"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

const defaultMaxFileSize = int64(100 * 1024 * 1024) // 100MB

// fileResponse 文件响应（嵌入 SysFile，附加运行时拼接的 url）
type fileResponse struct {
	model.SysFile
	URL string `json:"url"`
}

// md5Pattern MD5 格式：32 位十六进制（与 Python/Java 端一致，无效格式返回 B0404）
var md5Pattern = regexp.MustCompile(`^[0-9a-fA-F]{32}$`)

// attachFileURL 为单个 SysFile 附加运行时拼接的 URL
func (api *SysFileApi) attachFileURL(ctx context.Context, file model.SysFile) fileResponse {
	return fileResponse{SysFile: file, URL: api.fileService.GetURL(ctx, &file)}
}

// attachFileURLs 为 SysFile 列表附加运行时拼接的 URL
func (api *SysFileApi) attachFileURLs(ctx context.Context, files []model.SysFile) []fileResponse {
	result := make([]fileResponse, 0, len(files))
	for i := range files {
		result = append(result, api.attachFileURL(ctx, files[i]))
	}
	return result
}

type SysFileApi struct {
	fileService *fileservice.FileService
}

func NewSysFileApi(fileService *fileservice.FileService) *SysFileApi {
	return &SysFileApi{
		fileService: fileService,
	}
}

// ensureFileAccess 归属校验：管理员全量可见，普通用户仅可访问自己上传的文件（越权 B0407）
func ensureFileAccess(file *model.SysFile, userID int64, isAdmin bool) error {
	if isAdmin {
		return nil
	}
	if file.CreateBy != userID {
		return common.NewBizError(common.FILE_ACCESS_DENIED, "无权访问该文件")
	}
	return nil
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
		// B0402 与 Python/Java 端及文档错误码对齐
		_ = c.Error(common.NewBizError(common.FILE_TOO_LARGE, "文件大小超过限制"))
		return
	}

	// 3. 文件名安全校验，并回写 basename 化后的安全名（Service 层用其生成 name/扩展名）
	if err := validateFileName(fileHeader.Filename); err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, err.Error()))
		return
	}
	fileHeader.Filename = filepath.Base(strings.ReplaceAll(fileHeader.Filename, "\\", "/"))

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

	// 5. 调用 Service 上传（URL 不落库，运行时拼接；create_by 记录上传者供归属校验）
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	sysFile, err := api.fileService.UploadFile(ctx, fileHeader, reader, md5Hash, userID)
	if err != nil {
		_ = c.Error(err)
		return
	}

	// 响应中动态拼接 URL
	common.OkWithData(api.attachFileURL(ctx, sysFile), c)
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

	// 归属校验：普通用户仅可删除自己上传的文件
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	file, err := api.fileService.GetFileById(c.Request.Context(), fileId)
	if err != nil {
		_ = c.Error(err)
		return
	}
	if file.ID == 0 {
		_ = c.Error(common.NewBizError(common.FILE_NOT_FOUND, "文件不存在"))
		return
	}
	if err := ensureFileAccess(&file, userID, security.IsAdmin(c)); err != nil {
		_ = c.Error(err)
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
	// MD5 格式校验：32 位十六进制（T-FM-034/035：无效 MD5 返回 B0404，与 Python/Java 端一致）
	if !md5Pattern.MatchString(md5) {
		_ = c.Error(common.NewBizError(common.FILE_MD5_INVALID, "MD5格式无效"))
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
	common.OkWithData(api.attachFileURL(c.Request.Context(), *result), c)
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

	// 此前这里的 `err == nil` 即采用，连 pageSize=0/负值都放行（负值会让 LIMIT 子句消失、返回全量）
	pageNum, pageSize, ok := parsePagination(c)
	if !ok {
		return
	}
	keywords := c.Query("keywords")

	// 普通用户仅可见自己上传的文件，管理员全量
	var ownerID *int64
	if !security.IsAdmin(c) {
		userID, err := security.RequireUserID(c)
		if err != nil {
			_ = c.Error(err)
			return
		}
		ownerID = &userID
	}

	result, err := api.fileService.GetPage(ctx, pageNum, pageSize, keywords, ownerID)
	if err != nil {
		_ = c.Error(err)
		return
	}

	// 列表项附加运行时拼接的 URL
	if list, ok := result.List.([]model.SysFile); ok {
		result.List = api.attachFileURLs(ctx, list)
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

	// 文件不存在返回 B0401（与 Java/Python 端行为一致）
	if file.ID == 0 {
		_ = c.Error(common.NewBizError(common.FILE_NOT_FOUND, "文件不存在"))
		return
	}

	// 归属校验：普通用户仅可访问自己上传的文件
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := ensureFileAccess(&file, userID, security.IsAdmin(c)); err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithData(api.attachFileURL(c.Request.Context(), file), c)
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

	// Gin 的 *objectName 通配符会捕获带前导斜杠的路径，需去除以匹配数据库存储的 objectName
	objectName := strings.TrimPrefix(c.Param("objectName"), "/")
	if objectName == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "缺少objectName参数"))
		return
	}

	// 防止路径遍历攻击（对齐 Python 端校验）
	if strings.Contains(objectName, "..") || strings.HasPrefix(objectName, "/") || strings.Contains(objectName, "\\") {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "无效的文件路径"))
		return
	}

	file, err := api.fileService.GetFileByObjectName(ctx, objectName)
	if err != nil {
		_ = c.Error(common.WrapBizError(common.DATABASE_ERROR, "查询文件失败", err))
		return
	}
	if file == nil {
		_ = c.Error(common.NewBizError(common.FILE_NOT_FOUND, "文件不存在"))
		return
	}

	// 归属校验：普通用户仅可下载自己上传的文件
	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}
	if err := ensureFileAccess(file, userID, security.IsAdmin(c)); err != nil {
		_ = c.Error(err)
		return
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
	// 按扩展名推断 MIME 类型（对齐 Java/Python 端），推断不出时回退 application/octet-stream
	contentType := mime.TypeByExtension(filepath.Ext(filename))
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	c.Header("Content-Type", contentType)
	c.DataFromReader(200, -1, contentType, reader, nil)
}

// validateFileName 校验文件名安全性与长度（三端口径与 Python 端一致：
// 先 basename 化去除路径前缀，再拒绝非法字符与超长——sys_file.name/object_name 列宽 varchar(100)）
func validateFileName(fileName string) error {
	if fileName == "" {
		return fmt.Errorf("文件名不能为空")
	}
	fileName = filepath.Base(strings.ReplaceAll(fileName, "\\", "/"))
	if fileName == "" || fileName == "." || fileName == ".." {
		return fmt.Errorf("文件名不能为空")
	}
	if len([]rune(fileName)) > 100 {
		return fmt.Errorf("文件名过长")
	}
	// 禁止路径分隔符与特殊字符（与 Python 端 _UNSAFE_FILENAME_PATTERN 对齐）
	if strings.ContainsAny(fileName, `/:*?"<>|`) {
		return fmt.Errorf("文件名包含非法字符")
	}
	for _, ch := range fileName {
		if ch == '\\' || ch < 0x20 {
			return fmt.Errorf("文件名包含非法字符")
		}
	}
	ext := strings.TrimPrefix(filepath.Ext(fileName), ".")
	if len(ext) > 20 {
		return fmt.Errorf("文件扩展名过长")
	}
	return nil
}

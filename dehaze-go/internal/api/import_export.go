package api

import (
	"net/http"
	"net/url"
	"strconv"
	"strings"

	importexportservice "github.com/earthyzinc/dehaze-go/internal/service/import_export"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

type ImportExportApi struct {
	service *importexportservice.ImportExportService
}

func NewImportExportApi(service *importexportservice.ImportExportService) *ImportExportApi {
	return &ImportExportApi{service: service}
}

func (api *ImportExportApi) Export(c *gin.Context) {
	ctx := c.Request.Context()
	module := getModule(c)
	if module == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "模块名不能为空"))
		return
	}

	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	params := &importexportservice.ExportParams{
		Module: module,
		Query:  buildQueryFromRequest(c),
		Format: c.DefaultQuery("format", "excel"),
		Async:  parseBoolPtr(c.Query("async")),
		Fields: parseFields(c.Query("fields")),
		UserID: userID,
	}

	buf := &bufferedWriter{}
	result, err := api.service.Export(ctx, params, buf)
	if err != nil {
		_ = c.Error(err)
		return
	}

	if result != nil {
		common.OkWithDetailed(result, "导出任务已创建", c)
		return
	}
	api.writeExportFile(c, params.Format, module, buf)
}

func (api *ImportExportApi) ExportPost(c *gin.Context) {
	ctx := c.Request.Context()
	module := getModule(c)
	if module == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "模块名不能为空"))
		return
	}

	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	var body struct {
		Query   map[string]interface{} `json:"query"`
		Format  string                 `json:"format"`
		Async   *bool                  `json:"async"`
		Fields  []string               `json:"fields"`
		Options map[string]interface{} `json:"options"`
	}
	if err := c.ShouldBindJSON(&body); err != nil {
		_ = c.Error(err)
		return
	}

	if body.Format == "" {
		body.Format = "excel"
	}
	query := body.Query
	if query == nil {
		query = map[string]interface{}{}
	}
	if len(body.Options) > 0 {
		query["options"] = body.Options
	}

	params := &importexportservice.ExportParams{
		Module: module,
		Query:  query,
		Format: body.Format,
		Async:  body.Async,
		Fields: body.Fields,
		UserID: userID,
	}

	buf := &bufferedWriter{}
	result, err := api.service.Export(ctx, params, buf)
	if err != nil {
		_ = c.Error(err)
		return
	}

	if result != nil {
		common.OkWithDetailed(result, "导出任务已创建", c)
		return
	}
	api.writeExportFile(c, params.Format, module, buf)
}

func (api *ImportExportApi) Import(c *gin.Context) {
	ctx := c.Request.Context()
	module := getModule(c)
	if module == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "模块名不能为空"))
		return
	}

	userID, err := security.RequireUserID(c)
	if err != nil {
		_ = c.Error(err)
		return
	}

	fileHeader, err := c.FormFile("file")
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "文件上传失败"))
		return
	}
	file, err := fileHeader.Open()
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "打开上传文件失败"))
		return
	}
	defer file.Close()

	params := &importexportservice.ImportParams{
		Module:      module,
		File:        file,
		FileHeader:  fileHeader,
		Mode:        c.DefaultPostForm("mode", "all"),
		Async:       parseBoolPtr(c.PostForm("async")),
		ExtraParams: collectExtraParams(c),
		UserID:      userID,
	}

	result, err := api.service.Import(ctx, params)
	if err != nil {
		_ = c.Error(err)
		return
	}
	common.OkWithDetailed(result, "导入完成", c)
}

func (api *ImportExportApi) DownloadTemplate(c *gin.Context) {
	module := getModule(c)
	if module == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "模块名不能为空"))
		return
	}

	format := c.DefaultQuery("format", "excel")
	buf := &bufferedWriter{}
	if err := api.service.DownloadTemplate(buf, module, format); err != nil {
		_ = c.Error(err)
		return
	}

	fileExt := "xlsx"
	contentType := "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
	if strings.EqualFold(format, "csv") {
		fileExt = "csv"
		contentType = "text/csv"
	}
	fileName := module + "_template." + fileExt
	c.Header("Content-Description", "File Transfer")
	c.Header("Content-Disposition", "attachment; filename*=UTF-8''"+url.PathEscape(fileName))
	c.Header("Content-Type", contentType)
	c.Header("Content-Length", strconv.Itoa(len(buf.data)))
	c.Data(http.StatusOK, contentType, buf.data)
}

func (api *ImportExportApi) writeExportFile(c *gin.Context, format, module string, buf *bufferedWriter) {
	fileExt := "xlsx"
	contentType := "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
	if strings.EqualFold(format, "csv") {
		fileExt = "csv"
		contentType = "text/csv"
	}
	fileName := module + "_export." + fileExt
	c.Header("Content-Description", "File Transfer")
	c.Header("Content-Disposition", "attachment; filename*=UTF-8''"+url.PathEscape(fileName))
	c.Header("Content-Type", contentType)
	c.Header("Content-Length", strconv.Itoa(len(buf.data)))
	c.Data(http.StatusOK, contentType, buf.data)
}

func getModule(c *gin.Context) string {
	if m, exists := c.Get("importExportModule"); exists {
		if s, ok := m.(string); ok && s != "" {
			return s
		}
	}
	return c.Param("module")
}

func buildQueryFromRequest(c *gin.Context) map[string]interface{} {
	query := make(map[string]interface{})
	for k, v := range c.Request.URL.Query() {
		if len(v) == 0 {
			continue
		}
		if k == "async" || k == "format" || k == "fields" {
			continue
		}
		query[k] = v[0]
	}
	return query
}

func parseBoolPtr(s string) *bool {
	if s == "" {
		return nil
	}
	b := s == "true" || s == "1"
	return &b
}

func parseFields(s string) []string {
	if s == "" {
		return nil
	}
	return strings.Split(s, ",")
}

func collectExtraParams(c *gin.Context) map[string]interface{} {
	extra := make(map[string]interface{})
	for k, v := range c.Request.PostForm {
		if len(v) == 0 {
			continue
		}
		if k == "file" || k == "mode" || k == "async" {
			continue
		}
		extra[k] = v[0]
	}
	if len(extra) == 0 {
		return nil
	}
	return extra
}

type bufferedWriter struct {
	data []byte
}

func (b *bufferedWriter) Write(p []byte) (int, error) {
	b.data = append(b.data, p...)
	return len(p), nil
}

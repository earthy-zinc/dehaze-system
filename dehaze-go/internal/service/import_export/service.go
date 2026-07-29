package import_export

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/url"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/storage"
	"github.com/google/uuid"
	"go.uber.org/zap"
)

type ImportExportService struct {
	exportRegistry *ExportHandlerRegistry
	importRegistry *ImportHandlerRegistry
	fileGenerator  *FileGenerator
	templateMgr    *TemplateManager
	storage        storage.StorageService
	taskSvc        TaskService
	virusScanner   VirusScanner
	logger         *zap.Logger
}

type TaskService interface {
	CreateTask(ctx context.Context, taskType string, params interface{}, userID int64, idempotencyKey string) (*model.SysTask, error)
	GetTaskStatus(ctx context.Context, taskID string) (*model.SysTask, error)
	UpdateTaskProgress(ctx context.Context, taskID string, progress, current, total int) error
	UpdateTaskStatus(ctx context.Context, taskID string, status model.TaskStatus, errorMessage string) error
	UpdateTaskResult(ctx context.Context, taskID string, result string, expiresAt time.Time) error
	IsCancelled(ctx context.Context, taskID string) bool
}

func NewImportExportService(
	exportRegistry *ExportHandlerRegistry,
	importRegistry *ImportHandlerRegistry,
	fileGenerator *FileGenerator,
	templateMgr *TemplateManager,
	storage storage.StorageService,
	taskSvc TaskService,
	virusScanner VirusScanner,
	logger *zap.Logger,
) *ImportExportService {
	return &ImportExportService{
		exportRegistry: exportRegistry,
		importRegistry: importRegistry,
		fileGenerator:  fileGenerator,
		templateMgr:    templateMgr,
		storage:        storage,
		taskSvc:        taskSvc,
		virusScanner:   virusScanner,
		logger:         logger,
	}
}

type ExportParams struct {
	Module string
	Query  map[string]interface{}
	Format string
	Async  *bool
	Fields []string
	UserID int64
}

type ImportParams struct {
	Module      string
	File        multipart.File
	FileHeader  *multipart.FileHeader
	Mode        string
	Async       *bool
	ExtraParams map[string]interface{}
	UserID      int64
}

func (s *ImportExportService) Export(ctx context.Context, p *ExportParams, w io.Writer) (interface{}, error) {
	format := p.Format
	if format == "" {
		format = "excel"
	} else {
		format = strings.ToLower(format)
	}

	handler, err := s.exportRegistry.GetHandler(p.Module)
	if err != nil {
		return nil, err
	}

	count := handler.EstimateCount(p.Query)
	if count > int64(MaxRows) {
		return nil, common.NewBizError(common.EXPORT_ROWS_EXCEED_LIMIT,
			fmt.Sprintf("导出行数 %d 超出限制 %d", count, MaxRows))
	}

	shouldAsync := p.Async != nil && *p.Async
	if p.Async == nil {
		shouldAsync = count > int64(SyncThreshold)
	}

	if shouldAsync {
		return s.createExportTask(ctx, p.Module, p.Query, format, p.Fields, count, p.UserID)
	}

	if err := s.writeSyncExport(ctx, handler, p.Query, format, p.Fields, w); err != nil {
		return nil, err
	}
	return nil, nil
}

func (s *ImportExportService) writeSyncExport(ctx context.Context, handler ExportHandler, query map[string]interface{}, format string, fields []string, w io.Writer) error {
	direct := handler.UseDirectExport()
	exportCtx := &ExportContext{
		Module:         handler.GetModule(),
		Format:         format,
		SelectedFields: fields,
		QueryParams:    query,
		OutputStream:   w,
		TotalCount:     handler.EstimateCount(query),
		Async:          false,
		Ctx:            ctx,
	}

	if direct {
		return handler.Export(exportCtx, NoopProgressCallback{})
	}

	fieldConfigs := FilterFields(handler.GetFieldConfigs(), fields)
	if len(fieldConfigs) == 0 {
		return common.NewBizError(common.PARAM_ERROR, "无可导出字段")
	}

	provider := handler.GetDataProvider(exportCtx)
	if strings.EqualFold(format, "csv") {
		return s.fileGenerator.WriteCsv(w, fieldConfigs, provider)
	}
	return s.fileGenerator.WriteExcel(w, fieldConfigs, provider)
}

func (s *ImportExportService) createExportTask(ctx context.Context, module string, query map[string]interface{}, format string, fields []string, count int64, userID int64) (interface{}, error) {
	taskParams := map[string]interface{}{
		"module": module,
		"format": format,
		"fields": fields,
		"query":  query,
	}
	taskType := module + "_export"
	task, err := s.taskSvc.CreateTask(ctx, taskType, taskParams, userID, "")
	if err != nil {
		return nil, err
	}
	return ExportTaskResult{TaskID: task.TaskID, Status: int8(model.TaskStatusPending), EstimatedCount: count}, nil
}

func (s *ImportExportService) Import(ctx context.Context, p *ImportParams) (interface{}, error) {
	if err := s.validateUploadFile(p.FileHeader); err != nil {
		return nil, err
	}

	handler, err := s.importRegistry.GetHandler(p.Module)
	if err != nil {
		return nil, err
	}

	if p.Mode == "" {
		p.Mode = "all"
	}

	rows, err := s.parseRows(p.File, p.FileHeader.Filename, handler.GetDynamicFieldConfigs())
	if err != nil {
		return nil, err
	}
	rowCount := len(rows)
	if rowCount == 0 {
		return nil, common.NewBizError(common.IMPORT_FILE_EMPTY, "上传文件为空或无数据行")
	}
	if rowCount > MaxRows {
		return nil, common.NewBizError(common.IMPORT_ROWS_EXCEED_LIMIT,
			fmt.Sprintf("导入行数 %d 超出限制 %d", rowCount, MaxRows))
	}

	shouldAsync := p.Async != nil && *p.Async
	if p.Async == nil {
		shouldAsync = rowCount > SyncThreshold
	}

	if shouldAsync {
		if seeker, ok := p.File.(io.Seeker); ok {
			_, _ = seeker.Seek(0, io.SeekStart)
		}
		return s.createImportTask(ctx, p)
	}

	return s.executeSyncImportWithRows(ctx, handler, p, rows)
}

func (s *ImportExportService) executeSyncImportWithRows(ctx context.Context, handler ImportHandler, p *ImportParams, rows []map[string]interface{}) (ImportResultVO, error) {
	options := ImportOptions{Mode: p.Mode, ExtraParams: p.ExtraParams}
	result := handler.ImportBatch(rows, options, NoopProgressCallback{})

	var errorReportURL *string
	if result.FailureCount > 0 && len(result.Errors) > 0 {
		objectName := fmt.Sprintf("imports/%s_errors.xlsx", uuid.New().String())
		urlVal, genErr := s.generateErrorReport(ctx, objectName, result.Errors)
		if genErr != nil {
			s.logger.Warn("生成错误报告失败", zap.Error(genErr))
		} else {
			errorReportURL = &urlVal
		}
	}

	errors := result.Errors
	if errors == nil {
		errors = []ImportError{}
	}

	return ImportResultVO{
		TotalRows:      result.TotalRows,
		SuccessCount:   result.SuccessCount,
		FailureCount:   result.FailureCount,
		SkippedCount:   result.SkippedCount,
		Errors:         errors,
		ErrorReportUrl: errorReportURL,
	}, nil
}

func (s *ImportExportService) createImportTask(ctx context.Context, p *ImportParams) (ImportTaskResult, error) {
	objectName := fmt.Sprintf("temp/imports/%s/%s", uuid.New().String(), p.FileHeader.Filename)
	if err := s.storage.Upload(ctx, objectName, p.File, p.FileHeader.Size, p.FileHeader.Header.Get("Content-Type")); err != nil {
		return ImportTaskResult{}, common.WrapBizError(common.USER_UPLOAD_FILE_ERROR, "文件上传失败", err)
	}

	taskParams := map[string]interface{}{
		"module":        p.Module,
		"fileObjectName": objectName,
		"fileName":      p.FileHeader.Filename,
		"mode":          p.Mode,
		"extraParams":   p.ExtraParams,
	}
	taskType := p.Module + "_import"
	task, err := s.taskSvc.CreateTask(ctx, taskType, taskParams, p.UserID, "")
	if err != nil {
		return ImportTaskResult{}, err
	}
	return ImportTaskResult{TaskID: task.TaskID, Status: int8(model.TaskStatusPending)}, nil
}

func (s *ImportExportService) ExecuteAsyncExport(ctx context.Context, task *model.SysTask, params map[string]interface{}, callback ProgressCallback) {
	module, _ := params["module"].(string)
	format, _ := params["format"].(string)
	if format == "" {
		format = "excel"
	}
	fieldsRaw, _ := params["fields"].([]interface{})
	fields := make([]string, 0, len(fieldsRaw))
	for _, f := range fieldsRaw {
		if v, ok := f.(string); ok {
			fields = append(fields, v)
		}
	}
	query, _ := params["query"].(map[string]interface{})

	handler, err := s.exportRegistry.GetHandler(module)
	if err != nil {
		_ = s.taskSvc.UpdateTaskStatus(ctx, task.TaskID, model.TaskStatusFailed, "导出失败: "+err.Error())
		return
	}

	buf := &bytes.Buffer{}
	exportCtx := &ExportContext{
		TaskID:         task.TaskID,
		Module:         module,
		Format:         format,
		SelectedFields: fields,
		QueryParams:    query,
		OutputStream:   buf,
		TotalCount:     handler.EstimateCount(query),
		Async:          true,
		Ctx:            ctx,
	}

	callback.UpdateProgress(0, int(minInt64(exportCtx.TotalCount, int64(2147483647))), "开始导出")

	if handler.UseDirectExport() {
		if err := handler.Export(exportCtx, callback); err != nil {
			s.logger.Error("异步导出失败", zap.String("taskId", task.TaskID), zap.Error(err))
			_ = s.taskSvc.UpdateTaskStatus(ctx, task.TaskID, model.TaskStatusFailed, "导出失败: "+err.Error())
			return
		}
	} else {
		fieldConfigs := FilterFields(handler.GetFieldConfigs(), fields)
		if len(fieldConfigs) == 0 {
			_ = s.taskSvc.UpdateTaskStatus(ctx, task.TaskID, model.TaskStatusFailed, "无可导出字段")
			return
		}

		baseProvider := handler.GetDataProvider(exportCtx)
		wrappedProvider := &progressProvider{
			provider: baseProvider,
			total:    exportCtx.TotalCount,
			callback: callback,
		}

		var writeErr error
		if strings.EqualFold(format, "csv") {
			writeErr = s.fileGenerator.WriteCsv(buf, fieldConfigs, wrappedProvider)
		} else {
			writeErr = s.fileGenerator.WriteExcel(buf, fieldConfigs, wrappedProvider)
		}
		if writeErr != nil {
			s.logger.Error("异步导出失败", zap.String("taskId", task.TaskID), zap.Error(writeErr))
			_ = s.taskSvc.UpdateTaskStatus(ctx, task.TaskID, model.TaskStatusFailed, "导出失败: "+writeErr.Error())
			return
		}
	}

	fileExt := "xlsx"
	contentType := "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
	if handler.UseDirectExport() {
		fileExt = "zip"
		contentType = "application/zip"
	} else if strings.EqualFold(format, "csv") {
		fileExt = "csv"
		contentType = "text/csv"
	}

	objectName := fmt.Sprintf("exports/%s.%s", task.TaskID, fileExt)
	if err := s.storage.Upload(ctx, objectName, bytes.NewReader(buf.Bytes()), int64(buf.Len()), contentType); err != nil {
		s.logger.Error("异步导出上传文件失败", zap.String("taskId", task.TaskID), zap.Error(err))
		_ = s.taskSvc.UpdateTaskStatus(ctx, task.TaskID, model.TaskStatusFailed, "上传导出文件失败: "+err.Error())
		return
	}

	downloadURL, err := s.storage.GetURL(ctx, objectName)
	if err != nil {
		downloadURL = objectName
	}

	expiresAt := time.Now().Add(ResultFileExpireDays * 24 * time.Hour)
	_ = s.taskSvc.UpdateTaskResult(ctx, task.TaskID, downloadURL, expiresAt)

	s.logger.Info("异步导出完成",
		zap.String("taskId", task.TaskID),
		zap.String("module", module),
		zap.String("downloadUrl", downloadURL))
}

func (s *ImportExportService) ExecuteAsyncImport(ctx context.Context, task *model.SysTask, params map[string]interface{}, callback ProgressCallback) {
	module, _ := params["module"].(string)
	fileObjectName, _ := params["fileObjectName"].(string)
	fileName, _ := params["fileName"].(string)
	mode, _ := params["mode"].(string)
	if mode == "" {
		mode = "all"
	}
	extraParams, _ := params["extraParams"].(map[string]interface{})

	handler, err := s.importRegistry.GetHandler(module)
	if err != nil {
		_ = s.taskSvc.UpdateTaskStatus(ctx, task.TaskID, model.TaskStatusFailed, "导入失败: "+err.Error())
		return
	}

	reader, err := s.storage.Download(ctx, fileObjectName)
	if err != nil {
		s.logger.Error("异步导入下载文件失败", zap.String("taskId", task.TaskID), zap.Error(err))
		_ = s.taskSvc.UpdateTaskStatus(ctx, task.TaskID, model.TaskStatusFailed, "下载导入文件失败: "+err.Error())
		return
	}
	defer reader.Close()

	rows, err := s.parseRows(reader, fileName, handler.GetDynamicFieldConfigs())
	if err != nil {
		s.logger.Error("异步导入解析文件失败", zap.String("taskId", task.TaskID), zap.Error(err))
		_ = s.taskSvc.UpdateTaskStatus(ctx, task.TaskID, model.TaskStatusFailed, "导入失败: "+err.Error())
		return
	}

	options := ImportOptions{Mode: mode, ExtraParams: extraParams}
	result := handler.ImportBatch(rows, options, callback)

	var errorReportObjectName string
	if result.FailureCount > 0 && len(result.Errors) > 0 {
		objName := fmt.Sprintf("imports/%s_errors.xlsx", task.TaskID)
		urlVal, genErr := s.generateErrorReport(ctx, objName, result.Errors)
		if genErr != nil {
			s.logger.Warn("生成错误报告失败", zap.String("taskId", task.TaskID), zap.Error(genErr))
		} else {
			errorReportObjectName = urlVal
		}
	}

	resultJSON := s.buildImportResultJSON(result, errorReportObjectName)
	expiresAt := time.Now().Add(ResultFileExpireDays * 24 * time.Hour)
	_ = s.taskSvc.UpdateTaskResult(ctx, task.TaskID, resultJSON, expiresAt)

	s.logger.Info("异步导入完成",
		zap.String("taskId", task.TaskID),
		zap.String("module", module),
		zap.Int("success", result.SuccessCount),
		zap.Int("failure", result.FailureCount))
}

func (s *ImportExportService) buildImportResultJSON(result ImportResult, errorReportURL string) string {
	m := map[string]interface{}{
		"totalRows":    result.TotalRows,
		"successCount": result.SuccessCount,
		"failureCount": result.FailureCount,
		"skippedCount": result.SkippedCount,
		"errors":       result.Errors,
	}
	if errorReportURL != "" {
		m["errorReportUrl"] = errorReportURL
	}
	bytes, _ := json.Marshal(m)
	return string(bytes)
}

func (s *ImportExportService) generateErrorReport(ctx context.Context, objectName string, errors []ImportError) (string, error) {
	buf := &bytes.Buffer{}
	if err := s.fileGenerator.WriteErrorReport(buf, errors); err != nil {
		return "", err
	}
	if err := s.storage.Upload(ctx, objectName, bytes.NewReader(buf.Bytes()), int64(buf.Len()), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"); err != nil {
		return "", err
	}
	url, err := s.storage.GetURL(ctx, objectName)
	if err != nil {
		return objectName, nil
	}
	return url, nil
}

func (s *ImportExportService) validateUploadFile(fileHeader *multipart.FileHeader) error {
	if fileHeader == nil {
		return common.NewBizError(common.IMPORT_FILE_EMPTY, "上传文件为空")
	}
	if fileHeader.Size == 0 {
		return common.NewBizError(common.IMPORT_FILE_EMPTY, "上传文件为空")
	}
	if fileHeader.Size > MaxImportFileSize {
		return common.NewBizError(common.USER_UPLOAD_FILE_SIZE_EXCEEDS,
			fmt.Sprintf("文件大小 %d 超出限制 %d", fileHeader.Size, MaxImportFileSize))
	}
	name := fileHeader.Filename
	if name == "" {
		return common.NewBizError(common.USER_UPLOAD_FILE_TYPE_NOT_MATCH, "文件名为空")
	}
	lower := strings.ToLower(name)
	if !strings.HasSuffix(lower, ".xlsx") && !strings.HasSuffix(lower, ".xls") && !strings.HasSuffix(lower, ".csv") {
		return common.NewBizError(common.USER_UPLOAD_FILE_TYPE_NOT_MATCH, "不支持的文件类型: "+name)
	}
	return nil
}

func (s *ImportExportService) countRows(file multipart.File, fileName string, fields []ImportFieldConfig) (int, error) {
	count := 0
	err := s.fileGenerator.Parse(file, fileName, fields, func(int, map[string]interface{}) {
		count++
	})
	if err != nil {
		return 0, err
	}
	return count, nil
}

func (s *ImportExportService) parseRows(reader io.Reader, fileName string, fields []ImportFieldConfig) ([]map[string]interface{}, error) {
	rows := make([]map[string]interface{}, 0)
	err := s.fileGenerator.Parse(reader, fileName, fields, func(rowNum int, row map[string]interface{}) {
		rows = append(rows, row)
	})
	if err != nil {
		return nil, err
	}
	return rows, nil
}

func (s *ImportExportService) EncodeFilename(fileName string) string {
	return url.QueryEscape(fileName)
}

func (s *ImportExportService) DownloadTemplate(w io.Writer, module string, format string) error {
	handler, err := s.importRegistry.GetHandler(module)
	if err != nil {
		return err
	}
	return s.templateMgr.GenerateTemplate(w, handler, format)
}

type progressProvider struct {
	provider ExportDataProvider
	total    int64
	callback ProgressCallback
}

func (p *progressProvider) FetchBatch(pageNum, pageSize int) [][]interface{} {
	batch := p.provider.FetchBatch(pageNum, pageSize)
	if len(batch) > 0 {
		processed := pageNum * pageSize
		total := int(minInt64(p.total, int64(2147483647)))
		if total > 0 {
			processedInt := processed
			if processedInt > total {
				processedInt = total
			}
			p.callback.UpdateProgress(processedInt, total, fmt.Sprintf("导出中: %d/%d", processedInt, total))
		}
	}
	return batch
}

func minInt64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

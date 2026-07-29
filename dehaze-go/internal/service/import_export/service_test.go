package import_export

import (
	"bytes"
	"context"
	"errors"
	"io"
	"mime/multipart"
	"strings"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/storage"
	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
)

type configurableExportHandler struct {
	module  string
	count   int64
	direct  bool
	provide [][]interface{}
}

func (h *configurableExportHandler) GetModule() string { return h.module }
func (h *configurableExportHandler) EstimateCount(map[string]interface{}) int64 {
	return h.count
}
func (h *configurableExportHandler) Export(*ExportContext, ProgressCallback) error { return nil }
func (h *configurableExportHandler) GetFieldConfigs() []ExportFieldConfig {
	return []ExportFieldConfig{
		{Field: "username", Label: "用户名", Order: 1},
		{Field: "nickname", Label: "昵称", Order: 2},
	}
}
func (h *configurableExportHandler) GetDynamicFieldConfigs(map[string]interface{}) []ExportFieldConfig {
	return h.GetFieldConfigs()
}
func (h *configurableExportHandler) GetDataProvider(*ExportContext) ExportDataProvider {
	return &sliceDataProvider{rows: h.provide}
}
func (h *configurableExportHandler) UseDirectExport() bool { return h.direct }

type configurableImportHandler struct {
	module  string
	result  ImportResult
	fields  []ImportFieldConfig
	sample  []map[string]interface{}
	called  int
	lastRow []map[string]interface{}
}

func (h *configurableImportHandler) GetModule() string { return h.module }
func (h *configurableImportHandler) GetFieldConfigs() []ImportFieldConfig {
	if h.fields != nil {
		return h.fields
	}
	return []ImportFieldConfig{{Field: "username", Label: "用户名", Required: true}}
}
func (h *configurableImportHandler) GetDynamicFieldConfigs() []ImportFieldConfig {
	return h.GetFieldConfigs()
}
func (h *configurableImportHandler) ImportBatch(rows []map[string]interface{}, options ImportOptions, callback ProgressCallback) ImportResult {
	h.called++
	h.lastRow = rows
	if h.result.TotalRows == 0 && len(rows) > 0 {
		h.result = ImportResult{TotalRows: len(rows), SuccessCount: len(rows)}
	}
	return h.result
}
func (h *configurableImportHandler) GetTemplateSampleData() []map[string]interface{} {
	if h.sample != nil {
		return h.sample
	}
	return []map[string]interface{}{{"username": "zhangsan"}}
}

type mockTaskService struct {
	createTaskFn      func(ctx context.Context, taskType string, params interface{}, userID int64, idempotencyKey string) (*model.SysTask, error)
	updateStatusFn    func(ctx context.Context, taskID string, status model.TaskStatus, errorMessage string) error
	createCalls       int
	lastTaskType      string
	lastParams        interface{}
	updateResultCalls int
	updateStatusCalls int
	lastStatus        model.TaskStatus
	lastErrorMessage  string
}

func (m *mockTaskService) CreateTask(ctx context.Context, taskType string, params interface{}, userID int64, idempotencyKey string) (*model.SysTask, error) {
	m.createCalls++
	m.lastTaskType = taskType
	m.lastParams = params
	if m.createTaskFn != nil {
		return m.createTaskFn(ctx, taskType, params, userID, idempotencyKey)
	}
	return &model.SysTask{TaskID: "task-001", Status: model.TaskStatusPending}, nil
}
func (m *mockTaskService) GetTaskStatus(ctx context.Context, taskID string) (*model.SysTask, error) {
	return nil, nil
}
func (m *mockTaskService) UpdateTaskProgress(ctx context.Context, taskID string, progress, current, total int) error {
	return nil
}
func (m *mockTaskService) UpdateTaskStatus(ctx context.Context, taskID string, status model.TaskStatus, errorMessage string) error {
	m.updateStatusCalls++
	m.lastStatus = status
	m.lastErrorMessage = errorMessage
	if m.updateStatusFn != nil {
		return m.updateStatusFn(ctx, taskID, status, errorMessage)
	}
	return nil
}
func (m *mockTaskService) UpdateTaskResult(ctx context.Context, taskID string, result string, expiresAt time.Time) error {
	m.updateResultCalls++
	return nil
}
func (m *mockTaskService) IsCancelled(ctx context.Context, taskID string) bool { return false }

type mockStorage struct {
	uploadFn       func(ctx context.Context, objectName string, reader io.Reader, size int64, contentType string) error
	downloadFn     func(ctx context.Context, objectName string) (io.ReadCloser, error)
	getURLFn       func(ctx context.Context, objectName string) (string, error)
	uploadCalls    int
	lastObjectName string
}

func (m *mockStorage) Upload(ctx context.Context, objectName string, reader io.Reader, size int64, contentType string) error {
	m.uploadCalls++
	m.lastObjectName = objectName
	if m.uploadFn != nil {
		return m.uploadFn(ctx, objectName, reader, size, contentType)
	}
	return nil
}
func (m *mockStorage) Download(ctx context.Context, objectName string) (io.ReadCloser, error) {
	if m.downloadFn != nil {
		return m.downloadFn(ctx, objectName)
	}
	return io.NopCloser(strings.NewReader("")), nil
}
func (m *mockStorage) Delete(ctx context.Context, objectName string) error { return nil }
func (m *mockStorage) Exists(ctx context.Context, objectName string) (bool, error) {
	return false, nil
}
func (m *mockStorage) GetURL(ctx context.Context, objectName string) (string, error) {
	if m.getURLFn != nil {
		return m.getURLFn(ctx, objectName)
	}
	return "http://minio/" + objectName, nil
}

var _ storage.StorageService = (*mockStorage)(nil)

func newTestService(t *testing.T, exportHandlers []ExportHandler, importHandlers []ImportHandler) (*ImportExportService, *mockTaskService, *mockStorage) {
	exportReg := NewExportHandlerRegistry(exportHandlers)
	importReg := NewImportHandlerRegistry(importHandlers)
	gen := NewFileGenerator()
	tmplMgr := NewTemplateManager(gen)
	taskSvc := &mockTaskService{}
	storageSvc := &mockStorage{}
	logger := zap.NewNop()
	svc := NewImportExportService(exportReg, importReg, gen, tmplMgr, storageSvc, taskSvc, NoOpVirusScanner{}, logger)
	return svc, taskSvc, storageSvc
}

type autoResetFile struct {
	data  []byte
	pos   int
	atEOF bool
}

func (f *autoResetFile) Read(p []byte) (int, error) {
	if f.atEOF {
		f.atEOF = false
		f.pos = 0
	}
	if f.pos >= len(f.data) {
		f.atEOF = true
		return 0, io.EOF
	}
	n := copy(p, f.data[f.pos:])
	f.pos += n
	return n, nil
}
func (f *autoResetFile) ReadAt(p []byte, off int64) (int, error) {
	if off >= int64(len(f.data)) {
		return 0, io.EOF
	}
	n := copy(p, f.data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}
func (f *autoResetFile) Seek(offset int64, whence int) (int64, error) {
	var newPos int64
	switch whence {
	case io.SeekStart:
		newPos = offset
	case io.SeekCurrent:
		newPos = int64(f.pos) + offset
	case io.SeekEnd:
		newPos = int64(len(f.data)) + offset
	default:
		return 0, errors.New("invalid whence")
	}
	if newPos < 0 {
		return 0, errors.New("negative position")
	}
	f.pos = int(newPos)
	f.atEOF = false
	return newPos, nil
}
func (f *autoResetFile) Close() error { return nil }

func makeFileHeader(t *testing.T, filename string, content []byte) (*multipart.FileHeader, multipart.File) {
	t.Helper()
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile("file", filename)
	assert.NoError(t, err)
	_, err = part.Write(content)
	assert.NoError(t, err)
	assert.NoError(t, writer.Close())

	reader := multipart.NewReader(body, writer.Boundary())
	form, err := reader.ReadForm(int64(len(content) + 1024))
	assert.NoError(t, err)
	fileHeader := form.File["file"][0]
	file, err := fileHeader.Open()
	assert.NoError(t, err)
	return fileHeader, file
}

func makeImportParams(t *testing.T, filename string, content []byte, module string) *ImportParams {
	fileHeader, _ := makeFileHeader(t, filename, content)
	return &ImportParams{
		Module:     module,
		File:       &autoResetFile{data: content},
		FileHeader: fileHeader,
		Mode:       "all",
	}
}

func TestExport_Sync_ReturnsNil(t *testing.T) {
	handler := &configurableExportHandler{module: "user", count: 100}
	svc, _, _ := newTestService(t, []ExportHandler{handler}, nil)

	buf := &bytes.Buffer{}
	result, err := svc.Export(context.Background(), &ExportParams{
		Module: "user", Query: map[string]interface{}{}, Format: "excel",
	}, buf)
	assert.NoError(t, err)
	assert.Nil(t, result)
	assert.Greater(t, buf.Len(), 0)
}

func TestExport_Async_ReturnsTaskResult(t *testing.T) {
	handler := &configurableExportHandler{module: "user", count: int64(SyncThreshold + 1)}
	svc, taskSvc, _ := newTestService(t, []ExportHandler{handler}, nil)

	result, err := svc.Export(context.Background(), &ExportParams{
		Module: "user", Query: map[string]interface{}{}, Format: "excel", UserID: 1,
	}, &bytes.Buffer{})
	assert.NoError(t, err)
	taskRes, ok := result.(ExportTaskResult)
	assert.True(t, ok)
	assert.Equal(t, "task-001", taskRes.TaskID)
	assert.Equal(t, "PENDING", taskRes.Status)
	assert.Equal(t, int64(SyncThreshold+1), taskRes.EstimatedCount)
	assert.Equal(t, 1, taskSvc.createCalls)
	assert.Equal(t, "user_export", taskSvc.lastTaskType)
}

func TestExport_RowsExceedLimit_ReturnsError(t *testing.T) {
	handler := &configurableExportHandler{module: "user", count: int64(MaxRows + 1)}
	svc, _, _ := newTestService(t, []ExportHandler{handler}, nil)

	_, err := svc.Export(context.Background(), &ExportParams{
		Module: "user", Query: map[string]interface{}{},
	}, &bytes.Buffer{})
	assert.Error(t, err)
	var bizErr *common.BizError
	if errors.As(err, &bizErr) {
		assert.Equal(t, common.EXPORT_ROWS_EXCEED_LIMIT, bizErr.Code())
	}
}

func TestExport_ModuleNotSupported_ReturnsError(t *testing.T) {
	svc, _, _ := newTestService(t, nil, nil)

	_, err := svc.Export(context.Background(), &ExportParams{
		Module: "unknown", Query: map[string]interface{}{},
	}, &bytes.Buffer{})
	assert.Error(t, err)
	var bizErr *common.BizError
	if errors.As(err, &bizErr) {
		assert.Equal(t, common.MODULE_IMPORT_NOT_SUPPORTED, bizErr.Code())
	}
}

func TestExport_ForceSync_OverridesThreshold(t *testing.T) {
	handler := &configurableExportHandler{module: "user", count: int64(SyncThreshold + 100)}
	svc, taskSvc, _ := newTestService(t, []ExportHandler{handler}, nil)

	asyncFalse := false
	buf := &bytes.Buffer{}
	result, err := svc.Export(context.Background(), &ExportParams{
		Module: "user", Query: map[string]interface{}{}, Async: &asyncFalse,
	}, buf)
	assert.NoError(t, err)
	assert.Nil(t, result)
	assert.Equal(t, 0, taskSvc.createCalls)
	assert.Greater(t, buf.Len(), 0)
}

func TestExport_ForceAsync_OverridesThreshold(t *testing.T) {
	handler := &configurableExportHandler{module: "user", count: 10}
	svc, taskSvc, _ := newTestService(t, []ExportHandler{handler}, nil)

	asyncTrue := true
	result, err := svc.Export(context.Background(), &ExportParams{
		Module: "user", Query: map[string]interface{}{}, Async: &asyncTrue, UserID: 1,
	}, &bytes.Buffer{})
	assert.NoError(t, err)
	assert.IsType(t, ExportTaskResult{}, result)
	assert.Equal(t, 1, taskSvc.createCalls)
}

func TestExport_CsvFormat(t *testing.T) {
	handler := &configurableExportHandler{module: "user", count: 10}
	svc, _, _ := newTestService(t, []ExportHandler{handler}, nil)

	buf := &bytes.Buffer{}
	_, err := svc.Export(context.Background(), &ExportParams{
		Module: "user", Query: map[string]interface{}{}, Format: "csv",
	}, buf)
	assert.NoError(t, err)
	assert.True(t, strings.HasPrefix(buf.String(), "\ufeff"))
}

func TestExport_DirectExport_CallsHandlerExport(t *testing.T) {
	handler := &configurableExportHandler{module: "user", count: 10, direct: true}
	svc, _, _ := newTestService(t, []ExportHandler{handler}, nil)

	buf := &bytes.Buffer{}
	_, err := svc.Export(context.Background(), &ExportParams{
		Module: "user", Query: map[string]interface{}{}, Format: "excel",
	}, buf)
	assert.NoError(t, err)
}

func TestExport_SelectedFields_FiltersFieldConfigs(t *testing.T) {
	handler := &configurableExportHandler{
		module:  "user",
		count:   10,
		provide: [][]interface{}{{"u1", "n1"}},
	}
	svc, _, _ := newTestService(t, []ExportHandler{handler}, nil)

	buf := &bytes.Buffer{}
	_, err := svc.Export(context.Background(), &ExportParams{
		Module: "user", Query: map[string]interface{}{}, Format: "csv",
		Fields: []string{"username"},
	}, buf)
	assert.NoError(t, err)
	content := strings.TrimPrefix(buf.String(), "\ufeff")
	lines := strings.Split(content, "\n")
	assert.Equal(t, []string{"用户名"}, parseCSVLine(lines[0]))
	assert.Equal(t, []string{"u1"}, parseCSVLine(lines[1]))
}

func TestImport_UnsupportedFileType_ReturnsError(t *testing.T) {
	svc, _, _ := newTestService(t, nil, nil)
	params := makeImportParams(t, "test.txt", []byte("hello"), "user")

	_, err := svc.Import(context.Background(), params)
	assert.Error(t, err)
	var bizErr *common.BizError
	if errors.As(err, &bizErr) {
		assert.Equal(t, common.USER_UPLOAD_FILE_TYPE_NOT_MATCH, bizErr.Code())
	}
}

func TestImport_FileSizeExceeds_ReturnsError(t *testing.T) {
	svc, _, _ := newTestService(t, nil, nil)
	largeContent := bytes.Repeat([]byte("0"), int(MaxImportFileSize)+1)
	params := makeImportParams(t, "test.xlsx", largeContent, "user")

	_, err := svc.Import(context.Background(), params)
	assert.Error(t, err)
	var bizErr *common.BizError
	if errors.As(err, &bizErr) {
		assert.Equal(t, common.USER_UPLOAD_FILE_SIZE_EXCEEDS, bizErr.Code())
	}
}

func TestImport_EmptyFile_ReturnsError(t *testing.T) {
	svc, _, _ := newTestService(t, nil, nil)
	params := makeImportParams(t, "test.csv", []byte{}, "user")

	_, err := svc.Import(context.Background(), params)
	assert.Error(t, err)
	var bizErr *common.BizError
	if errors.As(err, &bizErr) {
		assert.Equal(t, common.IMPORT_FILE_EMPTY, bizErr.Code())
	}
}

func TestImport_ModuleNotSupported_ReturnsError(t *testing.T) {
	svc, _, _ := newTestService(t, nil, nil)
	content := []byte("\ufeff用户名\nu1\n")
	params := makeImportParams(t, "test.csv", content, "unknown")

	_, err := svc.Import(context.Background(), params)
	assert.Error(t, err)
	var bizErr *common.BizError
	if errors.As(err, &bizErr) {
		assert.Equal(t, common.MODULE_IMPORT_NOT_SUPPORTED, bizErr.Code())
	}
}

func TestImport_Sync_ReturnsResultVO(t *testing.T) {
	handler := &configurableImportHandler{module: "user"}
	svc, _, _ := newTestService(t, nil, []ImportHandler{handler})
	content := []byte("\ufeff用户名\nu1\nu2\n")
	params := makeImportParams(t, "test.csv", content, "user")

	result, err := svc.Import(context.Background(), params)
	assert.NoError(t, err)
	resultVO, ok := result.(ImportResultVO)
	assert.True(t, ok)
	assert.Equal(t, 2, resultVO.TotalRows)
	assert.Equal(t, 2, resultVO.SuccessCount)
	assert.Equal(t, 0, resultVO.FailureCount)
	assert.Equal(t, 1, handler.called)
	assert.Len(t, handler.lastRow, 2)
}

func TestImport_PartialMode_WithFailures_GeneratesErrorReport(t *testing.T) {
	handler := &configurableImportHandler{
		module: "user",
		result: ImportResult{
			TotalRows:    2,
			SuccessCount: 1,
			FailureCount: 1,
			Errors:       []ImportError{{Row: 2, Message: "用户名已存在"}},
		},
	}
	svc, _, storageSvc := newTestService(t, nil, []ImportHandler{handler})
	content := []byte("\ufeff用户名\nu1\nu2\n")
	params := makeImportParams(t, "test.csv", content, "user")
	params.Mode = "partial"

	result, err := svc.Import(context.Background(), params)
	assert.NoError(t, err)
	resultVO := result.(ImportResultVO)
	assert.Equal(t, 1, resultVO.FailureCount)
	assert.NotNil(t, resultVO.ErrorReportUrl)
	assert.Equal(t, 1, storageSvc.uploadCalls)
}

func TestImport_Async_ReturnsTaskResult(t *testing.T) {
	handler := &configurableImportHandler{module: "user"}
	svc, taskSvc, _ := newTestService(t, nil, []ImportHandler{handler})
	rows := make([]string, 0, SyncThreshold+2)
	rows = append(rows, "用户名")
	for i := 0; i <= SyncThreshold; i++ {
		rows = append(rows, "u"+itoa(i))
	}
	content := []byte("\ufeff" + strings.Join(rows, "\n") + "\n")
	params := makeImportParams(t, "test.csv", content, "user")
	params.UserID = 1

	result, err := svc.Import(context.Background(), params)
	assert.NoError(t, err)
	taskRes, ok := result.(ImportTaskResult)
	assert.True(t, ok)
	assert.Equal(t, "task-001", taskRes.TaskID)
	assert.Equal(t, "PENDING", taskRes.Status)
	assert.Equal(t, 1, taskSvc.createCalls)
	assert.Equal(t, "user_import", taskSvc.lastTaskType)
}

func TestImport_RowsExceedLimit_ReturnsError(t *testing.T) {
	handler := &configurableImportHandler{module: "user"}
	svc, _, _ := newTestService(t, nil, []ImportHandler{handler})
	rows := make([]string, 0, MaxRows+2)
	rows = append(rows, "用户名")
	for i := 0; i <= MaxRows; i++ {
		rows = append(rows, "u"+itoa(i))
	}
	content := []byte("\ufeff" + strings.Join(rows, "\n") + "\n")
	params := makeImportParams(t, "test.csv", content, "user")

	_, err := svc.Import(context.Background(), params)
	assert.Error(t, err)
	var bizErr *common.BizError
	if errors.As(err, &bizErr) {
		assert.Equal(t, common.IMPORT_ROWS_EXCEED_LIMIT, bizErr.Code())
	}
}

func TestImport_ForceSync_OverridesThreshold(t *testing.T) {
	handler := &configurableImportHandler{module: "user"}
	svc, taskSvc, _ := newTestService(t, nil, []ImportHandler{handler})
	rows := make([]string, 0, SyncThreshold+2)
	rows = append(rows, "用户名")
	for i := 0; i <= SyncThreshold; i++ {
		rows = append(rows, "u"+itoa(i))
	}
	content := []byte("\ufeff" + strings.Join(rows, "\n") + "\n")
	params := makeImportParams(t, "test.csv", content, "user")
	asyncFalse := false
	params.Async = &asyncFalse

	result, err := svc.Import(context.Background(), params)
	assert.NoError(t, err)
	assert.IsType(t, ImportResultVO{}, result)
	assert.Equal(t, 0, taskSvc.createCalls)
}

func TestExecuteAsyncExport_Success(t *testing.T) {
	handler := &configurableExportHandler{
		module:  "user",
		count:   10,
		provide: [][]interface{}{{"u1"}, {"u2"}},
	}
	svc, taskSvc, storageSvc := newTestService(t, []ExportHandler{handler}, nil)

	task := &model.SysTask{TaskID: "task-export-001"}
	params := map[string]interface{}{
		"module": "user",
		"format": "excel",
		"fields": []interface{}{"username"},
		"query":  map[string]interface{}{},
	}
	svc.ExecuteAsyncExport(context.Background(), task, params, NoopProgressCallback{})

	assert.Equal(t, 1, storageSvc.uploadCalls)
	assert.True(t, strings.HasPrefix(storageSvc.lastObjectName, "exports/task-export-001."))
	assert.Equal(t, 1, taskSvc.updateResultCalls)
}

func TestExecuteAsyncExport_HandlerError_UpdatesFailed(t *testing.T) {
	svc, taskSvc, _ := newTestService(t, nil, nil)

	task := &model.SysTask{TaskID: "task-export-fail"}
	params := map[string]interface{}{
		"module": "unknown",
		"format": "excel",
	}
	svc.ExecuteAsyncExport(context.Background(), task, params, NoopProgressCallback{})
	assert.Equal(t, 1, taskSvc.updateStatusCalls)
	assert.Equal(t, model.TaskStatusFailed, taskSvc.lastStatus)
	assert.Contains(t, taskSvc.lastErrorMessage, "导出失败")
}

func TestExecuteAsyncImport_Success(t *testing.T) {
	handler := &configurableImportHandler{
		module: "user",
		result: ImportResult{TotalRows: 2, SuccessCount: 2},
	}
	svc, taskSvc, _ := newTestService(t, nil, []ImportHandler{handler})

	csvContent := "\ufeff用户名\nu1\nu2\n"
	task := &model.SysTask{TaskID: "task-import-001"}
	params := map[string]interface{}{
		"module":        "user",
		"fileObjectName": "temp/imports/abc.csv",
		"fileName":      "test.csv",
		"mode":          "all",
	}
	svc.storage = &mockStorage{
		downloadFn: func(ctx context.Context, objectName string) (io.ReadCloser, error) {
			return io.NopCloser(strings.NewReader(csvContent)), nil
		},
	}
	svc.ExecuteAsyncImport(context.Background(), task, params, NoopProgressCallback{})
	assert.Equal(t, 1, taskSvc.updateResultCalls)
	assert.Equal(t, 1, handler.called)
}

func TestExecuteAsyncImport_WithFailures_GeneratesErrorReport(t *testing.T) {
	handler := &configurableImportHandler{
		module: "user",
		result: ImportResult{
			TotalRows:    2,
			SuccessCount: 1,
			FailureCount: 1,
			Errors:       []ImportError{{Row: 2, Message: "用户名已存在"}},
		},
	}
	svc, taskSvc, storageSvc := newTestService(t, nil, []ImportHandler{handler})
	storageSvc.downloadFn = func(ctx context.Context, objectName string) (io.ReadCloser, error) {
		return io.NopCloser(strings.NewReader("\ufeff用户名\nu1\nu2\n")), nil
	}

	task := &model.SysTask{TaskID: "task-import-002"}
	params := map[string]interface{}{
		"module":        "user",
		"fileObjectName": "temp/imports/abc.csv",
		"fileName":      "test.csv",
		"mode":          "partial",
	}
	svc.ExecuteAsyncImport(context.Background(), task, params, NoopProgressCallback{})
	assert.Equal(t, 1, taskSvc.updateResultCalls)
	assert.Equal(t, 1, storageSvc.uploadCalls)
}

func TestExecuteAsyncImport_HandlerError_UpdatesFailed(t *testing.T) {
	svc, taskSvc, _ := newTestService(t, nil, nil)

	task := &model.SysTask{TaskID: "task-import-fail"}
	params := map[string]interface{}{
		"module":        "unknown",
		"fileObjectName": "temp/imports/abc.csv",
		"fileName":      "test.csv",
		"mode":          "all",
	}
	svc.ExecuteAsyncImport(context.Background(), task, params, NoopProgressCallback{})
	assert.Equal(t, 1, taskSvc.updateStatusCalls)
	assert.Equal(t, model.TaskStatusFailed, taskSvc.lastStatus)
}

func TestDownloadTemplate_Csv(t *testing.T) {
	handler := &configurableImportHandler{module: "user"}
	svc, _, _ := newTestService(t, nil, []ImportHandler{handler})

	buf := &bytes.Buffer{}
	err := svc.DownloadTemplate(buf, "user", "csv")
	assert.NoError(t, err)
	assert.True(t, strings.HasPrefix(buf.String(), "\ufeff"))
}

func TestDownloadTemplate_Excel(t *testing.T) {
	handler := &configurableImportHandler{module: "user"}
	svc, _, _ := newTestService(t, nil, []ImportHandler{handler})

	buf := &bytes.Buffer{}
	err := svc.DownloadTemplate(buf, "user", "excel")
	assert.NoError(t, err)
	assert.Greater(t, buf.Len(), 0)
}

func TestDownloadTemplate_ModuleNotSupported(t *testing.T) {
	svc, _, _ := newTestService(t, nil, nil)

	err := svc.DownloadTemplate(&bytes.Buffer{}, "unknown", "csv")
	assert.Error(t, err)
}

func TestEncodeFilename(t *testing.T) {
	svc, _, _ := newTestService(t, nil, nil)
	assert.Equal(t, "%E7%94%A8%E6%88%B7%E5%AF%BC%E5%87%BA.xlsx", svc.EncodeFilename("用户导出.xlsx"))
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	negative := n < 0
	if negative {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if negative {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

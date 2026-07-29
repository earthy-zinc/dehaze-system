package import_export

import (
	"context"
	"io"
	"time"
)

const (
	SyncThreshold        = 1000
	MaxRows              = 100000
	MaxImportFileSizeMB  = 20
	MaxImportFileSize    = int64(MaxImportFileSizeMB) * 1024 * 1024
	ResultFileExpireDays = 7
	BatchSize            = 1000
)

type ExportFieldConfig struct {
	Field      string
	Label      string
	Order      int
	DateFormat string
	DictType   string
	Hidden     bool
}

type ImportFieldConfig struct {
	Field        string
	Label        string
	Required     bool
	DateFormat   string
	DictType     string
	Regex        string
	MaxLength    int
	DefaultValue string
}

type ExportContext struct {
	TaskID         string
	Module         string
	Format         string
	SelectedFields []string
	QueryParams    map[string]interface{}
	OutputStream   io.Writer
	TotalCount     int64
	Async          bool
	Ctx            context.Context
}

type ExportDataProvider interface {
	FetchBatch(pageNum, pageSize int) [][]interface{}
}

type ImportOptions struct {
	Mode        string
	ExtraParams map[string]interface{}
}

func (o ImportOptions) IsPartialMode() bool {
	return o.Mode == "partial"
}

type ImportError struct {
	Row     int
	Field   string
	Message string
}

type ImportResult struct {
	TotalRows   int
	SuccessCount int
	FailureCount int
	SkippedCount int
	Errors       []ImportError
}

func NewImportResult(total, success, failure int, errors []ImportError) ImportResult {
	return ImportResult{
		TotalRows:    total,
		SuccessCount: success,
		FailureCount: failure,
		SkippedCount: 0,
		Errors:       errors,
	}
}

type ProgressCallback interface {
	UpdateProgress(current, total int, message string)
	IsCancelled() bool
}

type NoopProgressCallback struct{}

func (NoopProgressCallback) UpdateProgress(int, int, string) {}
func (NoopProgressCallback) IsCancelled() bool               { return false }

type ProgressCallbackFunc struct {
	UpdateProgressFn func(current, total int, message string)
	IsCancelledFn    func() bool
}

func (p ProgressCallbackFunc) UpdateProgress(current, total int, message string) {
	if p.UpdateProgressFn != nil {
		p.UpdateProgressFn(current, total, message)
	}
}

func (p ProgressCallbackFunc) IsCancelled() bool {
	if p.IsCancelledFn != nil {
		return p.IsCancelledFn()
	}
	return false
}

type TaskRef struct {
	TaskID  string
	Status  string
}

type ExportTaskResult struct {
	TaskID         string `json:"taskId"`
	Status         int8   `json:"status"`
	EstimatedCount int64  `json:"estimatedCount"`
}

type ImportTaskResult struct {
	TaskID string `json:"taskId"`
	Status int8   `json:"status"`
}

type ImportResultVO struct {
	TotalRows        int           `json:"totalRows"`
	SuccessCount     int           `json:"successCount"`
	FailureCount     int           `json:"failureCount"`
	SkippedCount     int           `json:"skippedCount"`
	Errors           []ImportError `json:"errors"`
	ErrorReportUrl   *string       `json:"errorReportUrl"`
}

type TaskUpdateInfo struct {
	TaskID      string
	Result      string
	ExpiresAt   time.Time
}

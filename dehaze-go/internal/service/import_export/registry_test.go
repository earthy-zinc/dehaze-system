package import_export

import (
	"errors"
	"testing"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/stretchr/testify/assert"
)

type stubExportHandler struct {
	module  string
	count   int64
	dynamic bool
}

func (h *stubExportHandler) GetModule() string { return h.module }
func (h *stubExportHandler) EstimateCount(map[string]interface{}) int64 {
	return h.count
}
func (h *stubExportHandler) Export(*ExportContext, ProgressCallback) error { return nil }
func (h *stubExportHandler) GetFieldConfigs() []ExportFieldConfig {
	return []ExportFieldConfig{{Field: "f", Label: "F", Order: 1}}
}
func (h *stubExportHandler) GetDynamicFieldConfigs(map[string]interface{}) []ExportFieldConfig {
	if h.dynamic {
		return []ExportFieldConfig{{Field: "dyn", Label: "动态", Order: 1}}
	}
	return nil
}
func (h *stubExportHandler) GetDataProvider(*ExportContext) ExportDataProvider {
	return &stubDataProvider{}
}
func (h *stubExportHandler) UseDirectExport() bool { return false }

type stubDataProvider struct{}

func (s *stubDataProvider) FetchBatch(int, int) [][]interface{} { return nil }

type stubImportHandler struct {
	module string
}

func (h *stubImportHandler) GetModule() string { return h.module }
func (h *stubImportHandler) GetFieldConfigs() []ImportFieldConfig {
	return []ImportFieldConfig{{Field: "f", Label: "F", Required: true}}
}
func (h *stubImportHandler) GetDynamicFieldConfigs() []ImportFieldConfig {
	return []ImportFieldConfig{{Field: "f", Label: "F", Required: true}}
}
func (h *stubImportHandler) ImportBatch([]map[string]interface{}, ImportOptions, ProgressCallback) ImportResult {
	return ImportResult{}
}
func (h *stubImportHandler) GetTemplateSampleData() []map[string]interface{} {
	return []map[string]interface{}{{"f": "v"}}
}

func TestExportHandlerRegistry_RegisterAndGet(t *testing.T) {
	userH := &stubExportHandler{module: "user", count: 100}
	roleH := &stubExportHandler{module: "role", count: 50}
	registry := NewExportHandlerRegistry([]ExportHandler{userH, roleH})

	got, err := registry.GetHandler("user")
	assert.NoError(t, err)
	assert.Same(t, userH, got)

	got, err = registry.GetHandler("role")
	assert.NoError(t, err)
	assert.Same(t, roleH, got)
}

func TestExportHandlerRegistry_GetHandler_NotRegistered(t *testing.T) {
	registry := NewExportHandlerRegistry(nil)

	_, err := registry.GetHandler("unknown")
	assert.Error(t, err)
	var bizErr *common.BizError
	if errors.As(err, &bizErr) {
		assert.Equal(t, common.MODULE_IMPORT_NOT_SUPPORTED, bizErr.Code())
	}
}

func TestExportHandlerRegistry_DuplicateModulePanics(t *testing.T) {
	userH1 := &stubExportHandler{module: "user"}
	userH2 := &stubExportHandler{module: "user"}

	assert.Panics(t, func() {
		NewExportHandlerRegistry([]ExportHandler{userH1, userH2})
	})
}

func TestImportHandlerRegistry_RegisterAndGet(t *testing.T) {
	userH := &stubImportHandler{module: "user"}
	roleH := &stubImportHandler{module: "role"}
	registry := NewImportHandlerRegistry([]ImportHandler{userH, roleH})

	got, err := registry.GetHandler("user")
	assert.NoError(t, err)
	assert.Same(t, userH, got)

	got, err = registry.GetHandler("role")
	assert.NoError(t, err)
	assert.Same(t, roleH, got)
}

func TestImportHandlerRegistry_GetHandler_NotRegistered(t *testing.T) {
	registry := NewImportHandlerRegistry(nil)

	_, err := registry.GetHandler("unknown")
	assert.Error(t, err)
	var bizErr *common.BizError
	if errors.As(err, &bizErr) {
		assert.Equal(t, common.MODULE_IMPORT_NOT_SUPPORTED, bizErr.Code())
	}
}

func TestImportHandlerRegistry_DuplicateModulePanics(t *testing.T) {
	userH1 := &stubImportHandler{module: "user"}
	userH2 := &stubImportHandler{module: "user"}

	assert.Panics(t, func() {
		NewImportHandlerRegistry([]ImportHandler{userH1, userH2})
	})
}

func TestFilterFields_NoSelected_ReturnsVisibleSorted(t *testing.T) {
	all := []ExportFieldConfig{
		{Field: "b", Label: "B", Order: 2, Hidden: true},
		{Field: "c", Label: "C", Order: 3},
		{Field: "a", Label: "A", Order: 1},
	}
	result := FilterFields(all, nil)
	assert.Equal(t, []string{"a", "c"}, fieldNames(result))
}

func TestFilterFields_Selected_FiltersAndSorts(t *testing.T) {
	all := []ExportFieldConfig{
		{Field: "a", Label: "A", Order: 1},
		{Field: "b", Label: "B", Order: 2},
		{Field: "c", Label: "C", Order: 3},
	}
	result := FilterFields(all, []string{"c", "a"})
	assert.Equal(t, []string{"a", "c"}, fieldNames(result))
}

func TestFilterFields_Selected_ExcludesHidden(t *testing.T) {
	all := []ExportFieldConfig{
		{Field: "a", Label: "A", Order: 1},
		{Field: "b", Label: "B", Order: 2, Hidden: true},
	}
	result := FilterFields(all, []string{"a", "b"})
	assert.Equal(t, []string{"a"}, fieldNames(result))
}

func fieldNames(fields []ExportFieldConfig) []string {
	names := make([]string, 0, len(fields))
	for _, f := range fields {
		names = append(names, f.Field)
	}
	return names
}

func TestImportOptions_IsPartialMode(t *testing.T) {
	assert.True(t, ImportOptions{Mode: "partial"}.IsPartialMode())
	assert.False(t, ImportOptions{Mode: "all"}.IsPartialMode())
	assert.False(t, ImportOptions{Mode: ""}.IsPartialMode())
}

func TestNewImportResult(t *testing.T) {
	errs := []ImportError{{Row: 2, Message: "错误"}}
	result := NewImportResult(3, 2, 1, errs)
	assert.Equal(t, 3, result.TotalRows)
	assert.Equal(t, 2, result.SuccessCount)
	assert.Equal(t, 1, result.FailureCount)
	assert.Equal(t, 0, result.SkippedCount)
	assert.Len(t, result.Errors, 1)
}

func TestTaskTypeHelpers(t *testing.T) {
	assert.True(t, IsExportTaskType(TypeUserExport))
	assert.True(t, IsExportTaskType(TypeDatasetExport))
	assert.False(t, IsExportTaskType(TypeUserImport))

	assert.True(t, IsImportTaskType(TypeUserImport))
	assert.False(t, IsImportTaskType(TypeUserExport))

	assert.Equal(t, "user", ModuleFromTaskType(TypeUserExport))
	assert.Equal(t, "user", ModuleFromTaskType(TypeUserImport))
	assert.Equal(t, "dataset", ModuleFromTaskType(TypeDatasetExport))
	assert.Equal(t, "", ModuleFromTaskType("unknown"))

	assert.Equal(t, "export", TaskCategory(TypeUserExport))
	assert.Equal(t, "import", TaskCategory(TypeUserImport))
	assert.Equal(t, "", TaskCategory("unknown"))
}

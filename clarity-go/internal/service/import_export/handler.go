package import_export

type ExportHandler interface {
	GetModule() string
	EstimateCount(queryParams map[string]interface{}) int64
	Export(ctx *ExportContext, callback ProgressCallback) error
	GetFieldConfigs() []ExportFieldConfig
	GetDynamicFieldConfigs(queryParams map[string]interface{}) []ExportFieldConfig
	GetDataProvider(ctx *ExportContext) ExportDataProvider
	UseDirectExport() bool
}

type ImportHandler interface {
	GetModule() string
	GetFieldConfigs() []ImportFieldConfig
	GetDynamicFieldConfigs() []ImportFieldConfig
	ImportBatch(rows []map[string]interface{}, options ImportOptions, callback ProgressCallback) ImportResult
	GetTemplateSampleData() []map[string]interface{}
}

type BaseExportHandler struct{}

func (BaseExportHandler) Export(*ExportContext, ProgressCallback) error { return nil }
func (BaseExportHandler) UseDirectExport() bool                         { return false }
func (h BaseExportHandler) GetDynamicFieldConfigs(queryParams map[string]interface{}) []ExportFieldConfig {
	return nil
}

type BaseImportHandler struct{}

func (BaseImportHandler) GetDynamicFieldConfigs() []ImportFieldConfig {
	return nil
}
func (BaseImportHandler) GetTemplateSampleData() []map[string]interface{} {
	return nil
}

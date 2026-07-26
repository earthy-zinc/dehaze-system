package strategies

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/service/import_export"
)

type GenericImportStrategy struct {
	service *import_export.ImportExportService
}

func NewGenericImportStrategy(service *import_export.ImportExportService) *GenericImportStrategy {
	return &GenericImportStrategy{service: service}
}

func (s *GenericImportStrategy) GetTaskTypes() []string {
	return append([]string{}, import_export.ImportTaskTypes...)
}

func (s *GenericImportStrategy) Execute(ctx context.Context, task *model.SysTask, params map[string]interface{}, callback import_export.ProgressCallback) {
	s.service.ExecuteAsyncImport(ctx, task, params, callback)
}

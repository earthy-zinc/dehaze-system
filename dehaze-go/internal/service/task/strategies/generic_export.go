package strategies

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/service/import_export"
)

type GenericExportStrategy struct {
	service *import_export.ImportExportService
}

func NewGenericExportStrategy(service *import_export.ImportExportService) *GenericExportStrategy {
	return &GenericExportStrategy{service: service}
}

func (s *GenericExportStrategy) GetTaskTypes() []string {
	return append([]string{}, import_export.ExportTaskTypes...)
}

func (s *GenericExportStrategy) Execute(ctx context.Context, task *model.SysTask, params map[string]interface{}, callback import_export.ProgressCallback) {
	s.service.ExecuteAsyncExport(ctx, task, params, callback)
}

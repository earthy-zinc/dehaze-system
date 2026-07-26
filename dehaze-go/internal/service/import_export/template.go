package import_export

import (
	"io"
	"strings"

	"github.com/earthyzinc/dehaze-go/pkg/common"
)

type TemplateManager struct {
	fileGenerator *FileGenerator
}

func NewTemplateManager(fileGenerator *FileGenerator) *TemplateManager {
	return &TemplateManager{fileGenerator: fileGenerator}
}

func (m *TemplateManager) GenerateTemplate(w io.Writer, handler ImportHandler, format string) error {
	fields := handler.GetFieldConfigs()
	if len(fields) == 0 {
		return common.NewBizError(common.PARAM_ERROR, "该模块无可导入字段")
	}
	sampleData := handler.GetTemplateSampleData()

	formatLower := strings.ToLower(format)
	if formatLower == "csv" {
		return m.fileGenerator.WriteTemplateCsv(w, fields, sampleData)
	}
	return m.fileGenerator.WriteTemplateExcel(w, fields, sampleData)
}

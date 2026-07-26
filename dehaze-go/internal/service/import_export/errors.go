package import_export

import (
	"fmt"

	"github.com/earthyzinc/dehaze-go/pkg/common"
)

func NewModuleNotSupportedError(operation, module string) error {
	code := common.MODULE_IMPORT_NOT_SUPPORTED
	if operation == "export" {
		code = common.MODULE_EXPORT_NOT_SUPPORTED
	}
	return common.NewBizError(code, fmt.Sprintf("模块 %s 不支持%s", module, operation))
}

func NewImportError(code *common.ResultCode, message string) error {
	return common.NewBizError(code, message)
}

const (
	TypeUserExport      = "user_export"
	TypeRoleExport      = "role_export"
	TypeDeptExport      = "dept_export"
	TypeMenuExport      = "menu_export"
	TypeDictExport      = "dict_export"
	TypeDatasetExport   = "dataset_export"
	TypeAlgorithmExport = "algorithm_export"

	TypeUserImport      = "user_import"
	TypeRoleImport      = "role_import"
	TypeDeptImport      = "dept_import"
	TypeMenuImport      = "menu_import"
	TypeDictImport      = "dict_import"
	TypeAlgorithmImport = "algorithm_import"
)

var ExportTaskTypes = []string{
	TypeUserExport, TypeRoleExport, TypeDeptExport, TypeMenuExport,
	TypeDictExport, TypeDatasetExport, TypeAlgorithmExport,
}

var ImportTaskTypes = []string{
	TypeUserImport, TypeRoleImport, TypeDeptImport, TypeMenuImport,
	TypeDictImport, TypeAlgorithmImport,
}

func IsExportTaskType(taskType string) bool {
	for _, t := range ExportTaskTypes {
		if t == taskType {
			return true
		}
	}
	return false
}

func IsImportTaskType(taskType string) bool {
	for _, t := range ImportTaskTypes {
		if t == taskType {
			return true
		}
	}
	return false
}

func ModuleFromTaskType(taskType string) string {
	if len(taskType) <= len("_export") && len(taskType) <= len("_import") {
		return ""
	}
	if IsExportTaskType(taskType) {
		return taskType[:len(taskType)-len("_export")]
	}
	if IsImportTaskType(taskType) {
		return taskType[:len(taskType)-len("_import")]
	}
	return ""
}

func TaskCategory(taskType string) string {
	if IsExportTaskType(taskType) {
		return "export"
	}
	if IsImportTaskType(taskType) {
		return "import"
	}
	return ""
}

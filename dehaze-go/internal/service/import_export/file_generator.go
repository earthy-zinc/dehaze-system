package import_export

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"io"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/xuri/excelize/v2"
)

const utf8BOM = "\ufeff"

type FileGenerator struct{}

func NewFileGenerator() *FileGenerator {
	return &FileGenerator{}
}

func (g *FileGenerator) WriteExcel(w io.Writer, fields []ExportFieldConfig, provider ExportDataProvider) error {
	visibleFields := filterVisibleFieldsSorted(fields)
	if len(visibleFields) == 0 {
		return common.NewBizError(common.PARAM_ERROR, "无可导出字段")
	}

	f := excelize.NewFile()
	defer func() {
		_ = f.Close()
	}()

	sheetName := "Sheet1"
	if err := f.SetSheetName("Sheet1", sheetName); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "设置工作表失败", err)
	}

	for i, field := range visibleFields {
		cell, err := excelize.CoordinatesToCellName(i+1, 1)
		if err != nil {
			return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "构造单元格坐标失败", err)
		}
		if err := f.SetCellValue(sheetName, cell, field.Label); err != nil {
			return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "设置表头失败", err)
		}
	}

	pageNum := 1
	rowIdx := 2
	for {
		batch := provider.FetchBatch(pageNum, BatchSize)
		if len(batch) == 0 {
			break
		}
		for _, row := range batch {
			for i := 0; i < len(visibleFields); i++ {
				cell, err := excelize.CoordinatesToCellName(i+1, rowIdx)
				if err != nil {
					return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "构造单元格坐标失败", err)
				}
				val := ""
				if i < len(row) && row[i] != nil {
					val = fmt.Sprintf("%v", row[i])
				}
				if err := f.SetCellValue(sheetName, cell, val); err != nil {
					return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "设置导出数据失败", err)
				}
			}
			rowIdx++
		}
		pageNum++
	}

	return f.Write(w)
}

func (g *FileGenerator) WriteCsv(w io.Writer, fields []ExportFieldConfig, provider ExportDataProvider) error {
	visibleFields := filterVisibleFieldsSorted(fields)
	if len(visibleFields) == 0 {
		return common.NewBizError(common.PARAM_ERROR, "无可导出字段")
	}

	if _, err := io.WriteString(w, utf8BOM); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "写入 BOM 失败", err)
	}

	cw := csv.NewWriter(w)
	defer cw.Flush()

	headers := make([]string, len(visibleFields))
	for i, f := range visibleFields {
		headers[i] = f.Label
	}
	if err := cw.Write(headers); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "写入 CSV 表头失败", err)
	}

	pageNum := 1
	for {
		batch := provider.FetchBatch(pageNum, BatchSize)
		if len(batch) == 0 {
			break
		}
		for _, row := range batch {
			record := make([]string, len(visibleFields))
			for i := 0; i < len(visibleFields); i++ {
				if i < len(row) && row[i] != nil {
					record[i] = fmt.Sprintf("%v", row[i])
				}
			}
			if err := cw.Write(record); err != nil {
				return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "写入 CSV 数据失败", err)
			}
		}
		cw.Flush()
		if err := cw.Error(); err != nil {
			return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "刷新 CSV 缓存失败", err)
		}
		pageNum++
	}
	return nil
}

func (g *FileGenerator) WriteTemplateExcel(w io.Writer, fields []ImportFieldConfig, sampleData []map[string]interface{}) error {
	f := excelize.NewFile()
	defer func() {
		_ = f.Close()
	}()

	sheetName := "导入模板"
	if err := f.SetSheetName("Sheet1", sheetName); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "设置工作表失败", err)
	}

	for i, field := range fields {
		cell, err := excelize.CoordinatesToCellName(i+1, 1)
		if err != nil {
			return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "构造单元格坐标失败", err)
		}
		label := field.Label
		if field.Required {
			label = label + "*"
		}
		if err := f.SetCellValue(sheetName, cell, label); err != nil {
			return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "设置模板表头失败", err)
		}
	}

	for r, sample := range sampleData {
		rowIdx := r + 2
		for i, field := range fields {
			cell, err := excelize.CoordinatesToCellName(i+1, rowIdx)
			if err != nil {
				return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "构造单元格坐标失败", err)
			}
			val := ""
			if v, ok := sample[field.Field]; ok && v != nil {
				val = fmt.Sprintf("%v", v)
			}
			if err := f.SetCellValue(sheetName, cell, val); err != nil {
				return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "设置模板示例数据失败", err)
			}
		}
	}

	return f.Write(w)
}

func (g *FileGenerator) WriteTemplateCsv(w io.Writer, fields []ImportFieldConfig, sampleData []map[string]interface{}) error {
	if _, err := io.WriteString(w, utf8BOM); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "写入 BOM 失败", err)
	}

	cw := csv.NewWriter(w)
	defer cw.Flush()

	headers := make([]string, len(fields))
	for i, f := range fields {
		label := f.Label
		if f.Required {
			label = label + "*"
		}
		headers[i] = label
	}
	if err := cw.Write(headers); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "写入 CSV 模板表头失败", err)
	}

	for _, sample := range sampleData {
		record := make([]string, len(fields))
		for i, field := range fields {
			if v, ok := sample[field.Field]; ok && v != nil {
				record[i] = fmt.Sprintf("%v", v)
			}
		}
		if err := cw.Write(record); err != nil {
			return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "写入 CSV 模板示例失败", err)
		}
	}
	cw.Flush()
	if err := cw.Error(); err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "刷新 CSV 缓存失败", err)
	}
	return nil
}

type RowConsumer func(rowNum int, row map[string]interface{})

func (g *FileGenerator) Parse(r io.Reader, fileName string, fields []ImportFieldConfig, consumer RowConsumer) error {
	lower := strings.ToLower(fileName)
	if strings.HasSuffix(lower, ".csv") {
		return g.parseCsv(r, fields, consumer)
	}
	if strings.HasSuffix(lower, ".xlsx") || strings.HasSuffix(lower, ".xls") {
		return g.parseExcel(r, fields, consumer)
	}
	return common.NewBizError(common.USER_UPLOAD_FILE_TYPE_NOT_MATCH, "不支持的文件类型: "+fileName)
}

func (g *FileGenerator) parseExcel(r io.Reader, fields []ImportFieldConfig, consumer RowConsumer) error {
	f, err := excelize.OpenReader(r)
	if err != nil {
		return common.WrapBizError(common.IMPORT_FILE_PARSE_ERROR, "打开 Excel 文件失败", err)
	}
	defer func() { _ = f.Close() }()

	sheets := f.GetSheetList()
	if len(sheets) == 0 {
		return common.NewBizError(common.IMPORT_FILE_EMPTY, "Excel 无工作表")
	}
	rows, err := f.GetRows(sheets[0])
	if err != nil {
		return common.WrapBizError(common.IMPORT_FILE_PARSE_ERROR, "读取 Excel 工作表失败", err)
	}
	if len(rows) == 0 {
		return common.NewBizError(common.IMPORT_FILE_EMPTY, "Excel 无数据行")
	}

	labelToField := make(map[string]ImportFieldConfig, len(fields))
	for _, f := range fields {
		labelToField[f.Label] = f
	}

	headers := rows[0]
	for rowNum, data := range rows[1:] {
		row := make(map[string]interface{})
		for i, label := range headers {
			if i >= len(data) {
				continue
			}
			fc, ok := labelToField[label]
			if !ok {
				continue
			}
			row[fc.Field] = data[i]
		}
		consumer(rowNum+2, row)
	}
	return nil
}

func (g *FileGenerator) parseCsv(r io.Reader, fields []ImportFieldConfig, consumer RowConsumer) error {
	cr := csv.NewReader(r)
	cr.LazyQuotes = true
	cr.FieldsPerRecord = -1

	records, err := cr.ReadAll()
	if err != nil {
		return common.WrapBizError(common.IMPORT_FILE_PARSE_ERROR, "CSV 文件解析失败", err)
	}
	if len(records) == 0 {
		return common.NewBizError(common.IMPORT_FILE_EMPTY, "CSV 无数据行")
	}

	headers := records[0]
	labelToField := make(map[string]ImportFieldConfig, len(fields))
	for _, f := range fields {
		labelToField[f.Label] = f
	}

	for i := 1; i < len(records); i++ {
		data := records[i]
		row := make(map[string]interface{})
		for j, label := range headers {
			if j >= len(data) {
				continue
			}
			fc, ok := labelToField[label]
			if !ok {
				continue
			}
			row[fc.Field] = data[j]
		}
		consumer(i+1, row)
	}
	return nil
}

func (g *FileGenerator) CountRows(r io.Reader, fileName string, fields []ImportFieldConfig) (int, error) {
	count := 0
	err := g.Parse(r, fileName, fields, func(int, map[string]interface{}) {
		count++
	})
	return count, err
}

func filterVisibleFieldsSorted(fields []ExportFieldConfig) []ExportFieldConfig {
	visible := make([]ExportFieldConfig, 0, len(fields))
	for _, f := range fields {
		if !f.Hidden {
			visible = append(visible, f)
		}
	}
	sort.Slice(visible, func(i, j int) bool {
		return visible[i].Order < visible[j].Order
	})
	return visible
}

func FilterFields(all []ExportFieldConfig, selected []string) []ExportFieldConfig {
	if len(selected) == 0 {
		return filterVisibleFieldsSorted(all)
	}
	selectedSet := make(map[string]struct{}, len(selected))
	for _, s := range selected {
		selectedSet[s] = struct{}{}
	}
	result := make([]ExportFieldConfig, 0, len(all))
	for _, f := range all {
		if f.Hidden {
			continue
		}
		if _, ok := selectedSet[f.Field]; ok {
			result = append(result, f)
		}
	}
	sort.Slice(result, func(i, j int) bool {
		return result[i].Order < result[j].Order
	})
	return result
}

func (g *FileGenerator) WriteErrorReport(w io.Writer, errors []ImportError) error {
	fields := []ImportFieldConfig{
		{Field: "row", Label: "行号", Required: true},
		{Field: "field", Label: "字段"},
		{Field: "message", Label: "错误信息", Required: true},
	}
	sampleData := make([]map[string]interface{}, 0, len(errors))
	for _, e := range errors {
		sampleData = append(sampleData, map[string]interface{}{
			"row":     e.Row,
			"field":   e.Field,
			"message": e.Message,
		})
	}
	return g.WriteTemplateExcel(w, fields, sampleData)
}

var _ = bytes.NewReader
var _ = utf8.RuneLen

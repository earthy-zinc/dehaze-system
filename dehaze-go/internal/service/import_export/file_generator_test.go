package import_export

import (
	"bytes"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/xuri/excelize/v2"
)

type sliceDataProvider struct {
	rows     [][]interface{}
	consumed bool
}

func (p *sliceDataProvider) FetchBatch(pageNum, pageSize int) [][]interface{} {
	if p.consumed {
		return nil
	}
	p.consumed = true
	return p.rows
}

func newGenerator() *FileGenerator {
	return NewFileGenerator()
}

func TestWriteExcel_HeadersAndRows(t *testing.T) {
	gen := newGenerator()
	fields := []ExportFieldConfig{
		{Field: "username", Label: "用户名", Order: 2},
		{Field: "nickname", Label: "昵称", Order: 1},
		{Field: "secret", Label: "密钥", Order: 3, Hidden: true},
	}
	provider := &sliceDataProvider{rows: [][]interface{}{
		{"n1", "u1", "s1"},
		{"n2", "u2", nil},
	}}

	buf := &bytes.Buffer{}
	err := gen.WriteExcel(buf, fields, provider)
	assert.NoError(t, err)

	f, err := excelize.OpenReader(buf)
	assert.NoError(t, err)
	defer f.Close()

	rows, err := f.GetRows("Sheet1")
	assert.NoError(t, err)
	assert.Equal(t, []string{"昵称", "用户名"}, rows[0])
	assert.Equal(t, []string{"n1", "u1"}, rows[1])
	assert.Equal(t, []string{"n2", "u2"}, rows[2])
}

func TestWriteExcel_EmptyFields_ReturnsError(t *testing.T) {
	gen := newGenerator()
	provider := &sliceDataProvider{}
	buf := &bytes.Buffer{}
	err := gen.WriteExcel(buf, []ExportFieldConfig{{Field: "x", Label: "X", Order: 1, Hidden: true}}, provider)
	assert.Error(t, err)
}

func TestWriteCsv_BOMAndHeaders(t *testing.T) {
	gen := newGenerator()
	fields := []ExportFieldConfig{
		{Field: "username", Label: "用户名", Order: 2},
		{Field: "nickname", Label: "昵称", Order: 1},
	}
	provider := &sliceDataProvider{rows: [][]interface{}{
		{"u1", "n1"},
	}}

	buf := &bytes.Buffer{}
	err := gen.WriteCsv(buf, fields, provider)
	assert.NoError(t, err)

	content := buf.String()
	assert.True(t, strings.HasPrefix(content, "\ufeff"))
	lines := strings.Split(strings.TrimPrefix(content, "\ufeff"), "\n")
	assert.Equal(t, []string{"昵称", "用户名"}, parseCSVLine(lines[0]))
}

func TestWriteCsv_NilValueBecomesEmpty(t *testing.T) {
	gen := newGenerator()
	fields := []ExportFieldConfig{
		{Field: "username", Label: "用户名", Order: 1},
		{Field: "nickname", Label: "昵称", Order: 2},
	}
	provider := &sliceDataProvider{rows: [][]interface{}{{nil, "n1"}}}

	buf := &bytes.Buffer{}
	err := gen.WriteCsv(buf, fields, provider)
	assert.NoError(t, err)

	content := strings.TrimPrefix(buf.String(), "\ufeff")
	lines := strings.Split(content, "\n")
	assert.Equal(t, []string{"", "n1"}, parseCSVLine(lines[1]))
}

func TestWriteTemplateExcel_RequiredMarker(t *testing.T) {
	gen := newGenerator()
	fields := []ImportFieldConfig{
		{Field: "username", Label: "用户名", Required: true},
		{Field: "nickname", Label: "昵称", Required: false},
	}
	samples := []map[string]interface{}{
		{"username": "u1", "nickname": "n1"},
	}

	buf := &bytes.Buffer{}
	err := gen.WriteTemplateExcel(buf, fields, samples)
	assert.NoError(t, err)

	f, err := excelize.OpenReader(buf)
	assert.NoError(t, err)
	defer f.Close()

	rows, err := f.GetRows("导入模板")
	assert.NoError(t, err)
	assert.Equal(t, []string{"用户名*", "昵称"}, rows[0])
	assert.Equal(t, []string{"u1", "n1"}, rows[1])
}

func TestWriteTemplateCsv_RequiredMarker(t *testing.T) {
	gen := newGenerator()
	fields := []ImportFieldConfig{
		{Field: "username", Label: "用户名", Required: true},
	}
	samples := []map[string]interface{}{{"username": "u1"}}

	buf := &bytes.Buffer{}
	err := gen.WriteTemplateCsv(buf, fields, samples)
	assert.NoError(t, err)

	content := strings.TrimPrefix(buf.String(), "\ufeff")
	lines := strings.Split(content, "\n")
	assert.Equal(t, []string{"用户名*"}, parseCSVLine(lines[0]))
	assert.Equal(t, []string{"u1"}, parseCSVLine(lines[1]))
}

func TestParseCsv_MapsLabelToField(t *testing.T) {
	gen := newGenerator()
	content := "用户名,昵称\nu1,n1\nu2,n2\n"
	fields := []ImportFieldConfig{
		{Field: "username", Label: "用户名", Required: true},
		{Field: "nickname", Label: "昵称"},
	}

	var rows []map[string]interface{}
	err := gen.Parse(strings.NewReader(content), "test.csv", fields, func(rowNum int, row map[string]interface{}) {
		row["__rowNum__"] = rowNum
		rows = append(rows, row)
	})
	assert.NoError(t, err)
	assert.Len(t, rows, 2)
	assert.Equal(t, "u1", rows[0]["username"])
	assert.Equal(t, "n1", rows[0]["nickname"])
	assert.Equal(t, 2, rows[0]["__rowNum__"])
	assert.Equal(t, "u2", rows[1]["username"])
	assert.Equal(t, 3, rows[1]["__rowNum__"])
}

func TestParseCsv_EmptyFile_ReturnsError(t *testing.T) {
	gen := newGenerator()
	fields := []ImportFieldConfig{{Field: "username", Label: "用户名", Required: true}}

	err := gen.Parse(strings.NewReader(""), "test.csv", fields, func(int, map[string]interface{}) {})
	assert.Error(t, err)
}

func TestParse_UnsupportedFileType_ReturnsError(t *testing.T) {
	gen := newGenerator()
	fields := []ImportFieldConfig{{Field: "username", Label: "用户名", Required: true}}

	err := gen.Parse(strings.NewReader("x"), "test.txt", fields, func(int, map[string]interface{}) {})
	assert.Error(t, err)
}

func TestParseExcel_MapsLabelToField(t *testing.T) {
	gen := newGenerator()
	f := excelize.NewFile()
	sheetName := "Sheet1"
	f.SetSheetRow(sheetName, "A1", &[]interface{}{"用户名", "昵称"})
	f.SetSheetRow(sheetName, "A2", &[]interface{}{"u1", "n1"})
	f.SetSheetRow(sheetName, "A3", &[]interface{}{"u2", "n2"})

	buf := &bytes.Buffer{}
	_ = f.Write(buf)
	_ = f.Close()

	fields := []ImportFieldConfig{
		{Field: "username", Label: "用户名", Required: true},
		{Field: "nickname", Label: "昵称"},
	}
	var rows []map[string]interface{}
	err := gen.Parse(buf, "test.xlsx", fields, func(rowNum int, row map[string]interface{}) {
		rows = append(rows, row)
	})
	assert.NoError(t, err)
	assert.Len(t, rows, 2)
	assert.Equal(t, "u1", rows[0]["username"])
	assert.Equal(t, "n1", rows[0]["nickname"])
}

func TestCountRows(t *testing.T) {
	gen := newGenerator()
	content := "用户名\nu1\nu2\nu3\n"
	fields := []ImportFieldConfig{{Field: "username", Label: "用户名", Required: true}}

	count, err := gen.CountRows(strings.NewReader(content), "test.csv", fields)
	assert.NoError(t, err)
	assert.Equal(t, 3, count)
}

func TestWriteErrorReport(t *testing.T) {
	gen := newGenerator()
	errs := []ImportError{
		{Row: 2, Field: "username", Message: "用户名已存在"},
		{Row: 3, Field: "", Message: "校验失败"},
	}

	buf := &bytes.Buffer{}
	err := gen.WriteErrorReport(buf, errs)
	assert.NoError(t, err)

	f, err := excelize.OpenReader(buf)
	assert.NoError(t, err)
	defer f.Close()

	rows, err := f.GetRows("导入模板")
	assert.NoError(t, err)
	assert.Equal(t, []string{"行号*", "字段", "错误信息*"}, rows[0])
	assert.Equal(t, []string{"2", "username", "用户名已存在"}, rows[1])
	assert.Equal(t, []string{"3", "", "校验失败"}, rows[2])
}

func parseCSVLine(line string) []string {
	line = strings.TrimRight(line, "\r")
	if line == "" {
		return nil
	}
	return strings.Split(line, ",")
}

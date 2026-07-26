package handlers

import (
	"context"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/service/import_export"
	"gorm.io/gorm"
)

type DictExportHandler struct {
	import_export.BaseExportHandler
	db *gorm.DB
}

func NewDictExportHandler(db *gorm.DB) *DictExportHandler {
	return &DictExportHandler{db: db}
}

func (h *DictExportHandler) GetModule() string { return "dict" }

func (h *DictExportHandler) EstimateCount(params map[string]interface{}) int64 {
	q := buildDictQuery(params, 1, 1)
	var count int64
	tx := h.db.Model(&model.SysDict{})
	if q.Keywords != "" {
		like := "%" + q.Keywords + "%"
		tx = tx.Where("name LIKE ?", like)
	}
	if q.TypeCode != "" {
		tx = tx.Where("type_code = ?", q.TypeCode)
	}
	tx.Count(&count)
	return count
}

func (h *DictExportHandler) GetFieldConfigs() []import_export.ExportFieldConfig {
	return []import_export.ExportFieldConfig{
		{Field: "typeCode", Label: "字典类型编码", Order: 1},
		{Field: "name", Label: "字典名称", Order: 2},
		{Field: "value", Label: "字典值", Order: 3},
		{Field: "sort", Label: "排序", Order: 4},
		{Field: "statusLabel", Label: "状态", Order: 5},
		{Field: "defaulted", Label: "是否默认", Order: 6},
		{Field: "remark", Label: "备注", Order: 7},
		{Field: "createTime", Label: "创建时间", Order: 8, DateFormat: "2006-01-02 15:04:05"},
	}
}

func (h *DictExportHandler) GetDataProvider(ctx *import_export.ExportContext) import_export.ExportDataProvider {
	return &dictExportProvider{db: h.db, ctx: ctx}
}

type dictExportProvider struct {
	db  *gorm.DB
	ctx *import_export.ExportContext
}

func (p *dictExportProvider) FetchBatch(pageNum, pageSize int) [][]interface{} {
	q := buildDictQuery(p.ctx.QueryParams, pageNum, pageSize)
	var dicts []model.SysDict
	tx := p.db.Model(&model.SysDict{})
	if q.Keywords != "" {
		like := "%" + q.Keywords + "%"
		tx = tx.Where("name LIKE ?", like)
	}
	if q.TypeCode != "" {
		tx = tx.Where("type_code = ?", q.TypeCode)
	}
	tx.Order("sort ASC, create_time DESC").Offset((pageNum - 1) * pageSize).Limit(pageSize).Find(&dicts)

	rows := make([][]interface{}, 0, len(dicts))
	for _, d := range dicts {
		statusLabel := "禁用"
		if d.Status == 1 {
			statusLabel = "启用"
		}
		defaultedLabel := "否"
		if d.Defaulted == 1 {
			defaultedLabel = "是"
		}
		createTimeStr := ""
		if !d.CreatedAt.IsZero() {
			createTimeStr = d.CreatedAt.Format("2006-01-02 15:04:05")
		}
		rows = append(rows, []interface{}{
			d.TypeCode,
			d.Name,
			d.Value,
			d.Sort,
			statusLabel,
			defaultedLabel,
			d.Remark,
			createTimeStr,
		})
	}
	return rows
}

func buildDictQuery(params map[string]interface{}, pageNum, pageSize int) *query.DictPageQuery {
	q := &query.DictPageQuery{PageNum: pageNum, PageSize: pageSize}
	if params == nil {
		return q
	}
	if v, ok := params["keywords"].(string); ok {
		q.Keywords = v
	}
	if v, ok := params["typeCode"].(string); ok {
		q.TypeCode = v
	}
	return q
}

type DictImportHandler struct {
	import_export.BaseImportHandler
	db *gorm.DB
}

func NewDictImportHandler(db *gorm.DB) *DictImportHandler {
	return &DictImportHandler{db: db}
}

func (h *DictImportHandler) GetModule() string { return "dict" }

func (h *DictImportHandler) GetFieldConfigs() []import_export.ImportFieldConfig {
	return []import_export.ImportFieldConfig{
		{Field: "typeCode", Label: "字典类型编码", Required: true, MaxLength: 50},
		{Field: "name", Label: "字典名称", Required: true, MaxLength: 50},
		{Field: "value", Label: "字典值", Required: true, MaxLength: 50},
		{Field: "sort", Label: "排序"},
		{Field: "statusLabel", Label: "状态(启用/禁用)"},
		{Field: "defaulted", Label: "是否默认(是/否)"},
		{Field: "remark", Label: "备注"},
	}
}

func (h *DictImportHandler) GetTemplateSampleData() []map[string]interface{} {
	return []map[string]interface{}{
		{
			"typeCode":    "gender",
			"name":        "男",
			"value":       "1",
			"sort":        "1",
			"statusLabel": "启用",
			"defaulted":   "否",
			"remark":      "",
		},
	}
}

func (h *DictImportHandler) ImportBatch(rows []map[string]interface{}, options import_export.ImportOptions, callback import_export.ProgressCallback) import_export.ImportResult {
	total := len(rows)
	successCount := 0
	failureCount := 0
	var errors []import_export.ImportError
	ctx := context.Background()

	for i, row := range rows {
		rowNum := i + 2
		if callback.IsCancelled() {
			break
		}

		typeCode := getAsString(row, "typeCode")
		if typeCode == "" {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "typeCode", Message: "字典类型编码为空"})
			continue
		}
		name := getAsString(row, "name")
		if name == "" {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "name", Message: "字典名称为空"})
			continue
		}
		value := getAsString(row, "value")
		if value == "" {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "value", Message: "字典值为空"})
			continue
		}

		var exists int64
		h.db.Model(&model.SysDict{}).Where("type_code = ? AND value = ?", typeCode, value).Count(&exists)
		if exists > 0 {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "value", Message: "同类型下字典值已存在: " + value})
			continue
		}

		sort := parseInteger(row, "sort", 0)
		status := parseStatus(row, "statusLabel", 1)
		defaulted := parseBoolInt(row, "defaulted", 0)

		now := time.Now()
		dict := model.SysDict{
			TypeCode:  typeCode,
			Name:      name,
			Value:     value,
			Sort:      sort,
			Status:    int8(status),
			Defaulted: int8(defaulted),
			Remark:    getAsString(row, "remark"),
		}
		dict.CreatedAt = now
		dict.UpdatedAt = now

		if err := h.db.WithContext(ctx).Create(&dict).Error; err != nil {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Message: fmt.Sprintf("保存字典失败: %v", err)})
			continue
		}

		successCount++
		callback.UpdateProgress(i+1, total, fmt.Sprintf("导入第 %d 行", rowNum))
	}

	return import_export.ImportResult{
		TotalRows:    total,
		SuccessCount: successCount,
		FailureCount: failureCount,
		Errors:       errors,
	}
}

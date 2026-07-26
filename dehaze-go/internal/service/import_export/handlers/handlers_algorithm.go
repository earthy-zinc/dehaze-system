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

const (
	algorithmStatusDraft = 0
)

func algorithmStatusLabel(status int8) string {
	switch status {
	case 0:
		return "草稿"
	case 1:
		return "测试中"
	case 2:
		return "待审核"
	case 3:
		return "已发布"
	case 4:
		return "已停用"
	case 5:
		return "已归档"
	}
	return ""
}

type AlgorithmExportHandler struct {
	import_export.BaseExportHandler
	db *gorm.DB
}

func NewAlgorithmExportHandler(db *gorm.DB) *AlgorithmExportHandler {
	return &AlgorithmExportHandler{db: db}
}

func (h *AlgorithmExportHandler) GetModule() string { return "algorithm" }

func (h *AlgorithmExportHandler) EstimateCount(params map[string]interface{}) int64 {
	q := buildAlgorithmQuery(params)
	var count int64
	tx := h.db.Model(&model.SysAlgorithm{})
	if q.Keywords != "" {
		like := "%" + q.Keywords + "%"
		tx = tx.Where("name LIKE ? OR type LIKE ?", like, like)
	}
	if q.Type != "" {
		tx = tx.Where("type = ?", q.Type)
	}
	if q.Status != nil {
		tx = tx.Where("status = ?", *q.Status)
	}
	tx.Count(&count)
	return count
}

func (h *AlgorithmExportHandler) GetFieldConfigs() []import_export.ExportFieldConfig {
	return []import_export.ExportFieldConfig{
		{Field: "name", Label: "算法名称", Order: 1},
		{Field: "parentId", Label: "父算法ID", Order: 2},
		{Field: "type", Label: "算法类型", Order: 3},
		{Field: "path", Label: "模型文件路径", Order: 4},
		{Field: "importPath", Label: "导入路径", Order: 5},
		{Field: "description", Label: "描述", Order: 6},
		{Field: "version", Label: "版本", Order: 7},
		{Field: "statusLabel", Label: "状态", Order: 8},
		{Field: "size", Label: "大小", Order: 9},
		{Field: "flops", Label: "FLOPs", Order: 10},
		{Field: "params", Label: "参数量", Order: 11},
	}
}

func (h *AlgorithmExportHandler) GetDataProvider(ctx *import_export.ExportContext) import_export.ExportDataProvider {
	return &algorithmExportProvider{db: h.db, ctx: ctx}
}

type algorithmExportProvider struct {
	db  *gorm.DB
	ctx *import_export.ExportContext
}

func (p *algorithmExportProvider) FetchBatch(pageNum, pageSize int) [][]interface{} {
	q := buildAlgorithmQuery(p.ctx.QueryParams)
	var algos []model.SysAlgorithm
	tx := p.db.Model(&model.SysAlgorithm{})
	if q.Keywords != "" {
		like := "%" + q.Keywords + "%"
		tx = tx.Where("name LIKE ? OR type LIKE ?", like, like)
	}
	if q.Type != "" {
		tx = tx.Where("type = ?", q.Type)
	}
	if q.Status != nil {
		tx = tx.Where("status = ?", *q.Status)
	}
	tx.Order("id ASC").Offset((pageNum - 1) * pageSize).Limit(pageSize).Find(&algos)

	rows := make([][]interface{}, 0, len(algos))
	for _, a := range algos {
		version := ""
		if a.Version != nil {
			version = *a.Version
		}
		rows = append(rows, []interface{}{
			a.Name,
			a.ParentID,
			a.Type,
			a.Path,
			a.ImportPath,
			a.Description,
			version,
			algorithmStatusLabel(a.Status),
			a.Size,
			a.Flops,
			a.Params,
		})
	}
	return rows
}

func buildAlgorithmQuery(params map[string]interface{}) *query.AlgorithmQuery {
	q := &query.AlgorithmQuery{}
	if params == nil {
		return q
	}
	if v, ok := params["keywords"].(string); ok {
		q.Keywords = v
	}
	if v, ok := params["type"].(string); ok {
		q.Type = v
	}
	if v, ok := params["status"]; ok {
		switch s := v.(type) {
		case float64:
			si := int8(s)
			q.Status = &si
		case int:
			si := int8(s)
			q.Status = &si
		case int8:
			q.Status = &s
		}
	}
	return q
}

type AlgorithmImportHandler struct {
	import_export.BaseImportHandler
	db *gorm.DB
}

func NewAlgorithmImportHandler(db *gorm.DB) *AlgorithmImportHandler {
	return &AlgorithmImportHandler{db: db}
}

func (h *AlgorithmImportHandler) GetModule() string { return "algorithm" }

func (h *AlgorithmImportHandler) GetFieldConfigs() []import_export.ImportFieldConfig {
	return []import_export.ImportFieldConfig{
		{Field: "name", Label: "算法名称", Required: true, MaxLength: 50},
		{Field: "type", Label: "算法类型", Required: true},
		{Field: "parentId", Label: "父算法ID(0为顶级)"},
		{Field: "path", Label: "模型文件路径"},
		{Field: "importPath", Label: "导入路径"},
		{Field: "description", Label: "描述"},
		{Field: "version", Label: "版本"},
	}
}

func (h *AlgorithmImportHandler) GetTemplateSampleData() []map[string]interface{} {
	return []map[string]interface{}{
		{
			"name":        "示例去雾算法",
			"type":        "image_dehaze",
			"parentId":    "0",
			"path":        "/models/example.pth",
			"importPath":  "algorithms.example",
			"description": "示例算法",
			"version":     "1.0.0",
		},
	}
}

func (h *AlgorithmImportHandler) ImportBatch(rows []map[string]interface{}, options import_export.ImportOptions, callback import_export.ProgressCallback) import_export.ImportResult {
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

		name := getAsString(row, "name")
		if name == "" {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "name", Message: "算法名称为空"})
			continue
		}
		typ := getAsString(row, "type")
		if typ == "" {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "type", Message: "算法类型为空"})
			continue
		}

		var exists int64
		h.db.Model(&model.SysAlgorithm{}).Where("name = ?", name).Count(&exists)
		if exists > 0 {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "name", Message: "算法名称已存在: " + name})
			continue
		}

		parentID := parseLong(row, "parentId", 0)
		versionStr := getAsString(row, "version")
		var version *string
		if versionStr != "" {
			version = &versionStr
		}

		now := time.Now()
		algo := model.SysAlgorithm{
			ParentID:    parentID,
			Type:        typ,
			Name:        name,
			Path:        getAsString(row, "path"),
			ImportPath:  getAsString(row, "importPath"),
			Description: getAsString(row, "description"),
			Version:     version,
			Status:      algorithmStatusDraft,
		}
		algo.CreatedAt = now
		algo.UpdatedAt = now

		if err := h.db.WithContext(ctx).Create(&algo).Error; err != nil {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Message: fmt.Sprintf("保存算法失败: %v", err)})
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

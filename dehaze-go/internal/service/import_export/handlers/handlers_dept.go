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

type DeptExportHandler struct {
	import_export.BaseExportHandler
	db *gorm.DB
}

func NewDeptExportHandler(db *gorm.DB) *DeptExportHandler {
	return &DeptExportHandler{db: db}
}

func (h *DeptExportHandler) GetModule() string { return "dept" }

func (h *DeptExportHandler) EstimateCount(params map[string]interface{}) int64 {
	q := buildDeptQuery(params)
	var count int64
	tx := h.db.Model(&model.SysDept{})
	if q.Keywords != "" {
		like := "%" + q.Keywords + "%"
		tx = tx.Where("name LIKE ?", like)
	}
	if q.Status != nil {
		tx = tx.Where("status = ?", *q.Status)
	}
	tx.Count(&count)
	return count
}

func (h *DeptExportHandler) GetFieldConfigs() []import_export.ExportFieldConfig {
	return []import_export.ExportFieldConfig{
		{Field: "name", Label: "部门名称", Order: 1},
		{Field: "parentId", Label: "父部门ID", Order: 2},
		{Field: "sort", Label: "排序", Order: 3},
		{Field: "statusLabel", Label: "状态", Order: 4},
		{Field: "createTime", Label: "创建时间", Order: 5, DateFormat: "2006-01-02 15:04:05"},
	}
}

func (h *DeptExportHandler) GetDataProvider(ctx *import_export.ExportContext) import_export.ExportDataProvider {
	return &deptExportProvider{db: h.db, ctx: ctx}
}

type deptExportProvider struct {
	db  *gorm.DB
	ctx *import_export.ExportContext
}

func (p *deptExportProvider) FetchBatch(pageNum, pageSize int) [][]interface{} {
	q := buildDeptQuery(p.ctx.QueryParams)
	var depts []model.SysDept
	tx := p.db.Model(&model.SysDept{})
	if q.Keywords != "" {
		like := "%" + q.Keywords + "%"
		tx = tx.Where("name LIKE ?", like)
	}
	if q.Status != nil {
		tx = tx.Where("status = ?", *q.Status)
	}
	tx.Order("sort ASC, id ASC").Offset((pageNum - 1) * pageSize).Limit(pageSize).Find(&depts)

	rows := make([][]interface{}, 0, len(depts))
	for _, d := range depts {
		statusLabel := "禁用"
		if d.Status == 1 {
			statusLabel = "启用"
		}
		createTimeStr := ""
		if !d.CreatedAt.IsZero() {
			createTimeStr = d.CreatedAt.Format("2006-01-02 15:04:05")
		}
		rows = append(rows, []interface{}{
			d.Name,
			d.ParentID,
			d.Sort,
			statusLabel,
			createTimeStr,
		})
	}
	return rows
}

func buildDeptQuery(params map[string]interface{}) *query.DeptQuery {
	q := &query.DeptQuery{}
	if v, ok := params["keywords"].(string); ok {
		q.Keywords = v
	}
	if v, ok := params["status"]; ok {
		switch s := v.(type) {
		case float64:
			si := int(s)
			q.Status = &si
		case int:
			q.Status = &s
		}
	}
	return q
}

type DeptImportHandler struct {
	import_export.BaseImportHandler
	db *gorm.DB
}

func NewDeptImportHandler(db *gorm.DB) *DeptImportHandler {
	return &DeptImportHandler{db: db}
}

func (h *DeptImportHandler) GetModule() string { return "dept" }

func (h *DeptImportHandler) GetFieldConfigs() []import_export.ImportFieldConfig {
	return []import_export.ImportFieldConfig{
		{Field: "name", Label: "部门名称", Required: true, MaxLength: 64},
		{Field: "parentId", Label: "父部门ID(0为顶级)"},
		{Field: "sort", Label: "排序"},
		{Field: "statusLabel", Label: "状态(启用/禁用)"},
	}
}

func (h *DeptImportHandler) GetTemplateSampleData() []map[string]interface{} {
	return []map[string]interface{}{
		{"name": "研发部", "parentId": "0", "sort": "1", "statusLabel": "启用"},
	}
}

func (h *DeptImportHandler) ImportBatch(rows []map[string]interface{}, options import_export.ImportOptions, callback import_export.ProgressCallback) import_export.ImportResult {
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
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "name", Message: "部门名称为空"})
			continue
		}

		parentID := parseLong(row, "parentId", 0)
		sort := parseInteger(row, "sort", 0)
		status := parseStatus(row, "statusLabel", 1)

		var exists int64
		h.db.Model(&model.SysDept{}).Where("name = ? AND parent_id = ?", name, parentID).Count(&exists)
		if exists > 0 {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "name", Message: "同级下部门名称已存在: " + name})
			continue
		}

		now := time.Now()
		dept := model.SysDept{
			Name:     name,
			ParentID: parentID,
			Sort:     sort,
			Status:   int8(status),
			Deleted:  0,
		}
		dept.CreatedAt = now
		dept.UpdatedAt = now

		if parentID > 0 {
			var parent model.SysDept
			if err := h.db.WithContext(ctx).Where("id = ?", parentID).First(&parent).Error; err == nil {
				dept.TreePath = parent.TreePath + fmt.Sprintf(",%d", parentID)
			}
		} else {
			dept.TreePath = "0"
		}

		if err := h.db.WithContext(ctx).Create(&dept).Error; err != nil {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Message: fmt.Sprintf("保存部门失败: %v", err)})
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

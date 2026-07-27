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

type RoleExportHandler struct {
	import_export.BaseExportHandler
	db *gorm.DB
}

func NewRoleExportHandler(db *gorm.DB) *RoleExportHandler {
	return &RoleExportHandler{db: db}
}

func (h *RoleExportHandler) GetModule() string { return "role" }

func (h *RoleExportHandler) EstimateCount(params map[string]interface{}) int64 {
	q := buildRoleQuery(params)
	var count int64
	tx := h.db.Model(&model.SysRole{})
	if q.Keywords != "" {
		like := "%" + q.Keywords + "%"
		tx = tx.Where("name LIKE ? OR code LIKE ?", like, like)
	}
	tx.Count(&count)
	return count
}

func (h *RoleExportHandler) GetFieldConfigs() []import_export.ExportFieldConfig {
	return []import_export.ExportFieldConfig{
		{Field: "name", Label: "角色名称", Order: 1},
		{Field: "code", Label: "角色编码", Order: 2},
		{Field: "sort", Label: "排序", Order: 3},
		{Field: "statusLabel", Label: "状态", Order: 4},
		{Field: "createTime", Label: "创建时间", Order: 5, DateFormat: "2006-01-02 15:04:05"},
	}
}

func (h *RoleExportHandler) GetDataProvider(ctx *import_export.ExportContext) import_export.ExportDataProvider {
	return &roleExportProvider{db: h.db, ctx: ctx}
}

type roleExportProvider struct {
	db  *gorm.DB
	ctx *import_export.ExportContext
}

func (p *roleExportProvider) FetchBatch(pageNum, pageSize int) [][]interface{} {
	q := buildRoleQuery(p.ctx.QueryParams)
	tx := p.db.Model(&model.SysRole{})
	if q.Keywords != "" {
		like := "%" + q.Keywords + "%"
		tx = tx.Where("name LIKE ? OR code LIKE ?", like, like)
	}
	var roles []model.SysRole
	tx.Order("id ASC").Offset((pageNum - 1) * pageSize).Limit(pageSize).Find(&roles)

	rows := make([][]interface{}, 0, len(roles))
	for _, r := range roles {
		statusLabel := "禁用"
		if r.Status == 1 {
			statusLabel = "启用"
		}
		createTimeStr := ""
		if !r.CreatedAt.IsZero() {
			createTimeStr = r.CreatedAt.Format("2006-01-02 15:04:05")
		}
		rows = append(rows, []interface{}{
			r.Name,
			r.Code,
			r.Sort,
			statusLabel,
			createTimeStr,
		})
	}
	return rows
}

func buildRoleQuery(params map[string]interface{}) *query.RolePageQuery {
	q := &query.RolePageQuery{}
	if v, ok := params["keywords"].(string); ok {
		q.Keywords = v
	}
	return q
}

type RoleImportHandler struct {
	import_export.BaseImportHandler
	db *gorm.DB
}

func NewRoleImportHandler(db *gorm.DB) *RoleImportHandler {
	return &RoleImportHandler{db: db}
}

func (h *RoleImportHandler) GetModule() string { return "role" }

func (h *RoleImportHandler) GetFieldConfigs() []import_export.ImportFieldConfig {
	return []import_export.ImportFieldConfig{
		{Field: "name", Label: "角色名称", Required: true, MaxLength: 64},
		{Field: "code", Label: "角色编码", Required: true, MaxLength: 32},
		{Field: "sort", Label: "排序"},
		{Field: "statusLabel", Label: "状态(启用/禁用)"},
	}
}

func (h *RoleImportHandler) GetTemplateSampleData() []map[string]interface{} {
	return []map[string]interface{}{
		{"name": "普通用户", "code": "user", "sort": "1", "statusLabel": "启用"},
	}
}

func (h *RoleImportHandler) ImportBatch(rows []map[string]interface{}, options import_export.ImportOptions, callback import_export.ProgressCallback) import_export.ImportResult {
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
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "name", Message: "角色名称为空"})
			continue
		}
		code := getAsString(row, "code")
		if code == "" {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "code", Message: "角色编码为空"})
			continue
		}

		var exists int64
		h.db.Model(&model.SysRole{}).Where("code = ?", code).Count(&exists)
		if exists > 0 {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "code", Message: "角色编码已存在: " + code})
			continue
		}

		sort := parseInteger(row, "sort", 0)
		status := parseStatus(row, "statusLabel", 1)

		now := time.Now()
		role := model.SysRole{
			Name:      name,
			Code:      code,
			Sort:      sort,
			Status:    int8(status),
			DataScope: 5,
			Deleted:   0,
		}
		role.CreatedAt = now
		role.UpdatedAt = now

		if err := h.db.WithContext(ctx).Create(&role).Error; err != nil {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Message: fmt.Sprintf("保存角色失败: %v", err)})
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

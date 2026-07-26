package handlers

import (
	"context"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/enum"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/service/import_export"
	"gorm.io/gorm"
)

type MenuExportHandler struct {
	import_export.BaseExportHandler
	db *gorm.DB
}

func NewMenuExportHandler(db *gorm.DB) *MenuExportHandler {
	return &MenuExportHandler{db: db}
}

func (h *MenuExportHandler) GetModule() string { return "menu" }

func (h *MenuExportHandler) EstimateCount(params map[string]interface{}) int64 {
	q := buildMenuQuery(params)
	var count int64
	tx := h.db.Model(&model.SysMenu{}).Where("1=1")
	if q.Keywords != "" {
		like := "%" + q.Keywords + "%"
		tx = tx.Where("name LIKE ?", like)
	}
	if q.Status != nil {
		tx = tx.Where("visible = ?", *q.Status)
	}
	tx.Count(&count)
	return count
}

func (h *MenuExportHandler) GetFieldConfigs() []import_export.ExportFieldConfig {
	return []import_export.ExportFieldConfig{
		{Field: "name", Label: "菜单名称", Order: 1},
		{Field: "parentId", Label: "父菜单ID", Order: 2},
		{Field: "typeLabel", Label: "类型", Order: 3},
		{Field: "path", Label: "路由路径", Order: 4},
		{Field: "component", Label: "组件路径", Order: 5},
		{Field: "perm", Label: "权限标识", Order: 6},
		{Field: "visible", Label: "是否可见", Order: 7},
		{Field: "sort", Label: "排序", Order: 8},
		{Field: "icon", Label: "图标", Order: 9},
		{Field: "redirect", Label: "跳转路径", Order: 10},
	}
}

func (h *MenuExportHandler) GetDataProvider(ctx *import_export.ExportContext) import_export.ExportDataProvider {
	return &menuExportProvider{db: h.db, ctx: ctx}
}

type menuExportProvider struct {
	db  *gorm.DB
	ctx *import_export.ExportContext
}

func (p *menuExportProvider) FetchBatch(pageNum, pageSize int) [][]interface{} {
	q := buildMenuQuery(p.ctx.QueryParams)
	var menus []model.SysMenu
	tx := p.db.Model(&model.SysMenu{})
	if q.Keywords != "" {
		like := "%" + q.Keywords + "%"
		tx = tx.Where("name LIKE ?", like)
	}
	if q.Status != nil {
		tx = tx.Where("visible = ?", *q.Status)
	}
	tx.Order("sort ASC, id ASC").Offset((pageNum - 1) * pageSize).Limit(pageSize).Find(&menus)

	rows := make([][]interface{}, 0, len(menus))
	for _, m := range menus {
		typeLabel := enum.GetMenuTypeName(int(m.Type))
		visibleLabel := "隐藏"
		if m.Visible == 1 {
			visibleLabel = "显示"
		}
		rows = append(rows, []interface{}{
			m.Name,
			m.ParentID,
			typeLabel,
			m.Path,
			m.Component,
			m.Perm,
			visibleLabel,
			m.Sort,
			m.Icon,
			m.Redirect,
		})
	}
	return rows
}

func buildMenuQuery(params map[string]interface{}) *query.MenuQuery {
	q := &query.MenuQuery{}
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

type MenuImportHandler struct {
	import_export.BaseImportHandler
	db *gorm.DB
}

func NewMenuImportHandler(db *gorm.DB) *MenuImportHandler {
	return &MenuImportHandler{db: db}
}

func (h *MenuImportHandler) GetModule() string { return "menu" }

func (h *MenuImportHandler) GetFieldConfigs() []import_export.ImportFieldConfig {
	return []import_export.ImportFieldConfig{
		{Field: "name", Label: "菜单名称", Required: true, MaxLength: 64},
		{Field: "parentId", Label: "父菜单ID(0为顶级)"},
		{Field: "typeLabel", Label: "类型(菜单/目录/外链/按钮)", Required: true},
		{Field: "path", Label: "路由路径"},
		{Field: "component", Label: "组件路径"},
		{Field: "perm", Label: "权限标识"},
		{Field: "visible", Label: "是否可见(显示/隐藏)"},
		{Field: "sort", Label: "排序"},
		{Field: "icon", Label: "图标"},
		{Field: "redirect", Label: "跳转路径"},
	}
}

func (h *MenuImportHandler) GetTemplateSampleData() []map[string]interface{} {
	return []map[string]interface{}{
		{
			"name":       "用户管理",
			"parentId":   "0",
			"typeLabel":  "菜单",
			"path":       "/system/user",
			"component":  "system/user/index",
			"perm":       "sys:user:list",
			"visible":    "显示",
			"sort":       "1",
			"icon":       "user",
			"redirect":   "",
		},
	}
}

func parseMenuType(label string) (int8, bool) {
	for v, name := range enum.MenuTypeNames {
		if name == label {
			return int8(v), true
		}
	}
	return 0, false
}

func parseVisible(row map[string]interface{}, key string, defaultValue int8) int8 {
	s := getAsString(row, key)
	if s == "" {
		return defaultValue
	}
	if s == "显示" {
		return 1
	}
	if s == "隐藏" {
		return 0
	}
	return defaultValue
}

func (h *MenuImportHandler) ImportBatch(rows []map[string]interface{}, options import_export.ImportOptions, callback import_export.ProgressCallback) import_export.ImportResult {
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
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "name", Message: "菜单名称为空"})
			continue
		}

		typeLabel := getAsString(row, "typeLabel")
		if typeLabel == "" {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "typeLabel", Message: "菜单类型为空"})
			continue
		}
		menuType, ok := parseMenuType(typeLabel)
		if !ok {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "typeLabel", Message: "菜单类型无效(应为 菜单/目录/外链/按钮): " + typeLabel})
			continue
		}

		parentID := parseLong(row, "parentId", 0)
		visible := parseVisible(row, "visible", 1)
		sort := parseInteger(row, "sort", 0)

		now := time.Now()
		menu := model.SysMenu{
			ParentID:  parentID,
			Name:      name,
			Type:      menuType,
			Path:      getAsString(row, "path"),
			Component: getAsString(row, "component"),
			Perm:      getAsString(row, "perm"),
			Visible:   visible,
			Sort:      sort,
			Icon:      getAsString(row, "icon"),
			Redirect:  getAsString(row, "redirect"),
		}
		menu.CreatedAt = now
		menu.UpdatedAt = now

		if parentID > 0 {
			var parent model.SysMenu
			if err := h.db.WithContext(ctx).Where("id = ?", parentID).First(&parent).Error; err == nil {
				menu.TreePath = parent.TreePath + fmt.Sprintf(",%d", parentID)
			}
		} else {
			menu.TreePath = "0"
		}

		if err := h.db.WithContext(ctx).Create(&menu).Error; err != nil {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Message: fmt.Sprintf("保存菜单失败: %v", err)})
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

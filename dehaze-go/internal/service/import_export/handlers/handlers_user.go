package handlers

import (
	"context"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/repository/dept"
	"github.com/earthyzinc/dehaze-go/internal/service/import_export"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/gorm"
)

type UserExportHandler struct {
	import_export.BaseExportHandler
	db *gorm.DB
}

func NewUserExportHandler(db *gorm.DB) *UserExportHandler {
	return &UserExportHandler{db: db}
}

func (h *UserExportHandler) GetModule() string { return "user" }

func (h *UserExportHandler) EstimateCount(params map[string]interface{}) int64 {
	q := buildUserQuery(params)
	var count int64
	tx := h.db.Model(&model.SysUser{}).Where("deleted = 0")
	if q.Keywords != "" {
		like := "%" + q.Keywords + "%"
		tx = tx.Where("(username LIKE ? OR nickname LIKE ? OR mobile LIKE ?)", like, like, like)
	}
	tx.Count(&count)
	return count
}

func (h *UserExportHandler) GetFieldConfigs() []import_export.ExportFieldConfig {
	return []import_export.ExportFieldConfig{
		{Field: "username", Label: "用户名", Order: 1},
		{Field: "nickname", Label: "昵称", Order: 2},
		{Field: "gender", Label: "性别", Order: 3, DictType: "gender"},
		{Field: "deptName", Label: "部门", Order: 4},
		{Field: "mobile", Label: "手机号", Order: 5},
		{Field: "email", Label: "邮箱", Order: 6},
		{Field: "status", Label: "状态", Order: 7, DictType: "user_status"},
		{Field: "createTime", Label: "创建时间", Order: 8, DateFormat: "2006-01-02 15:04:05"},
	}
}

func (h *UserExportHandler) GetDataProvider(ctx *import_export.ExportContext) import_export.ExportDataProvider {
	return &userExportProvider{db: h.db, ctx: ctx}
}

type userExportProvider struct {
	db  *gorm.DB
	ctx *import_export.ExportContext
}

func (p *userExportProvider) FetchBatch(pageNum, pageSize int) [][]interface{} {
	var users []model.SysUser
	q := buildUserQuery(p.ctx.QueryParams)
	tx := p.db.Model(&model.SysUser{}).Where("deleted = 0")
	if q.Keywords != "" {
		like := "%" + q.Keywords + "%"
		tx = tx.Where("username LIKE ? OR nickname LIKE ? OR mobile LIKE ?", like, like, like)
	}
	if q.Status != nil {
		tx = tx.Where("status = ?", *q.Status)
	}
	if q.DeptId != nil {
		tx = tx.Where("dept_id = ?", *q.DeptId)
	}
	if q.StartTime != "" {
		tx = tx.Where("created_at >= ?", q.StartTime)
	}
	if q.EndTime != "" {
		tx = tx.Where("created_at <= ?", q.EndTime)
	}
	tx.Order("id ASC").Offset((pageNum - 1) * pageSize).Limit(pageSize).Find(&users)

	deptIDs := make([]int64, 0, len(users))
	for _, u := range users {
		if u.DeptID > 0 {
			deptIDs = append(deptIDs, u.DeptID)
		}
	}
	deptNameMap := make(map[int64]string)
	if len(deptIDs) > 0 {
		var depts []model.SysDept
		p.db.Where("id IN ?", deptIDs).Find(&depts)
		for _, d := range depts {
			deptNameMap[d.ID] = d.Name
		}
	}

	rows := make([][]interface{}, 0, len(users))
	for _, u := range users {
		genderLabel := "未知"
		switch u.Gender {
		case 1:
			genderLabel = "男"
		case 2:
			genderLabel = "女"
		}
		statusLabel := "禁用"
		if u.Status == 1 {
			statusLabel = "启用"
		}
		createTimeStr := ""
		if !u.CreatedAt.IsZero() {
			createTimeStr = u.CreatedAt.Format("2006-01-02 15:04:05")
		}
		rows = append(rows, []interface{}{
			u.Username,
			u.Nickname,
			genderLabel,
			deptNameMap[u.DeptID],
			u.Mobile,
			u.Email,
			statusLabel,
			createTimeStr,
		})
	}
	return rows
}

func buildUserQuery(params map[string]interface{}) *query.UserPageQuery {
	q := &query.UserPageQuery{}
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
	if v, ok := params["deptId"]; ok {
		switch d := v.(type) {
		case float64:
			di := int64(d)
			q.DeptId = &di
		case int64:
			q.DeptId = &d
		}
	}
	if v, ok := params["startTime"].(string); ok {
		q.StartTime = v
	}
	if v, ok := params["endTime"].(string); ok {
		q.EndTime = v
	}
	return q
}

type UserImportHandler struct {
	import_export.BaseImportHandler
	db              *gorm.DB
	deptRepo        dept.IDeptRepository
	defaultPassword string
}

func NewUserImportHandler(db *gorm.DB, deptRepo dept.IDeptRepository, defaultPassword string) *UserImportHandler {
	return &UserImportHandler{db: db, deptRepo: deptRepo, defaultPassword: defaultPassword}
}

func (h *UserImportHandler) GetModule() string { return "user" }

func (h *UserImportHandler) GetDynamicFieldConfigs() []import_export.ImportFieldConfig {
	return h.GetFieldConfigs()
}

func (h *UserImportHandler) GetFieldConfigs() []import_export.ImportFieldConfig {
	return []import_export.ImportFieldConfig{
		{Field: "username", Label: "用户名", Required: true},
		{Field: "nickname", Label: "昵称", Required: true},
		{Field: "gender", Label: "性别"},
		{Field: "deptName", Label: "部门名称"},
		{Field: "mobile", Label: "手机号"},
		{Field: "email", Label: "邮箱"},
		{Field: "status", Label: "状态"},
	}
}

func (h *UserImportHandler) GetTemplateSampleData() []map[string]interface{} {
	return []map[string]interface{}{
		{"username": "user001", "nickname": "测试用户1", "gender": "男", "deptName": "研发部", "mobile": "13800138001", "email": "user001@example.com", "status": "启用"},
		{"username": "user002", "nickname": "测试用户2", "gender": "女", "deptName": "测试部", "mobile": "13800138002", "email": "user002@example.com", "status": "启用"},
	}
}

func (h *UserImportHandler) ImportBatch(rows []map[string]interface{}, options import_export.ImportOptions, callback import_export.ProgressCallback) import_export.ImportResult {
	total := len(rows)
	successCount := 0
	failureCount := 0
	var errors []import_export.ImportError

	deptNameSet := make(map[string]struct{})
	for _, row := range rows {
		if name, ok := row["deptName"].(string); ok && name != "" {
			deptNameSet[name] = struct{}{}
		}
	}
	deptNameList := make([]string, 0, len(deptNameSet))
	for name := range deptNameSet {
		deptNameList = append(deptNameList, name)
	}
	ctx := context.Background()
	deptIDMap := make(map[string]int64)
	if len(deptNameList) > 0 {
		m, err := h.deptRepo.FindIDsByNames(ctx, deptNameList)
		if err == nil {
			deptIDMap = m
		}
	}

	for i, row := range rows {
		rowNum := i + 2
		if callback.IsCancelled() {
			break
		}

		username, _ := row["username"].(string)
		nickname, _ := row["nickname"].(string)
		if username == "" || nickname == "" {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "username", Message: "用户名和昵称为必填项"})
			continue
		}

		var existingCount int64
		h.db.Model(&model.SysUser{}).Where("username = ?", username).Count(&existingCount)
		if existingCount > 0 {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "username", Message: "用户名已存在"})
			continue
		}

		var genderInt int8
		if gender, ok := row["gender"].(string); ok {
			switch gender {
			case "男":
				genderInt = 1
			case "女":
				genderInt = 2
			}
		}

		var status int8 = 1
		if statusStr, ok := row["status"].(string); ok {
			if statusStr == "禁用" || statusStr == "0" {
				status = 0
			}
		}

		var deptID int64
		if deptName, ok := row["deptName"].(string); ok && deptName != "" {
			deptID = deptIDMap[deptName]
		}

		mobile, _ := row["mobile"].(string)
		email, _ := row["email"].(string)

		hashedPassword, err := bcrypt.GenerateFromPassword([]byte(h.defaultPassword), bcrypt.DefaultCost)
		if err != nil {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "password", Message: "密码加密失败"})
			continue
		}

		now := time.Now()
		user := model.SysUser{
			Username: username,
			Nickname: nickname,
			Gender:   genderInt,
			DeptID:   deptID,
			Mobile:   mobile,
			Email:    email,
			Status:   status,
			Password: string(hashedPassword),
		}
		user.CreatedAt = now
		user.UpdatedAt = now

		if err := h.db.Create(&user).Error; err != nil {
			failureCount++
			errors = append(errors, import_export.ImportError{Row: rowNum, Field: "username", Message: fmt.Sprintf("保存用户失败: %v", err)})
			continue
		}

		successCount++
		callback.UpdateProgress(i+1, total, fmt.Sprintf("导入中: %d/%d", i+1, total))
	}

	return import_export.ImportResult{
		TotalRows:    total,
		SuccessCount: successCount,
		FailureCount: failureCount,
		Errors:       errors,
	}
}

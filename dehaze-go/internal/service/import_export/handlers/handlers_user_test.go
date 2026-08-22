package handlers

import (
	"context"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
	"github.com/earthyzinc/dehaze-go/internal/repository/dept"
	"github.com/earthyzinc/dehaze-go/internal/service/import_export"
	"github.com/glebarez/sqlite"
	"github.com/stretchr/testify/assert"
	"gorm.io/gorm"
)

type mockDeptRepo struct {
	namesToIDs map[string]int64
	err        error
	called     int
	lastNames  []string
}

func (m *mockDeptRepo) FindByID(ctx context.Context, id int64) (*model.SysDept, error) {
	return nil, nil
}
func (m *mockDeptRepo) FindAll(ctx context.Context, q *query.DeptQuery) ([]model.SysDept, error) {
	return nil, nil
}
func (m *mockDeptRepo) FindByParentID(ctx context.Context, parentID int64) ([]model.SysDept, error) {
	return nil, nil
}
func (m *mockDeptRepo) FindIDByName(ctx context.Context, name string) (int64, error) {
	return 0, nil
}
func (m *mockDeptRepo) Create(ctx context.Context, d *model.SysDept) error { return nil }
func (m *mockDeptRepo) Update(ctx context.Context, d *model.SysDept) error { return nil }
func (m *mockDeptRepo) Delete(ctx context.Context, ids []int64) error      { return nil }
func (m *mockDeptRepo) HasChildren(ctx context.Context, id int64) (bool, error) {
	return false, nil
}
func (m *mockDeptRepo) HasUsers(ctx context.Context, deptID int64) (bool, error) {
	return false, nil
}
func (m *mockDeptRepo) HasUsersInBatch(ctx context.Context, deptIDs []int64) (map[int64]bool, error) {
	return nil, nil
}
func (m *mockDeptRepo) FindIDsByNames(ctx context.Context, names []string) (map[string]int64, error) {
	m.called++
	m.lastNames = names
	if m.err != nil {
		return nil, m.err
	}
	return m.namesToIDs, nil
}
func (m *mockDeptRepo) GetOptions(ctx context.Context) ([]read.Option, error) { return nil, nil }
func (m *mockDeptRepo) GetFormData(ctx context.Context, deptID int64) (*bo.DeptFormBO, error) {
	return nil, nil
}
func (m *mockDeptRepo) GetSubDeptIDs(ctx context.Context, deptID int64) ([]int64, error) {
	return nil, nil
}

var _ dept.IDeptRepository = (*mockDeptRepo)(nil)

func newTestDB(t *testing.T) *gorm.DB {
	t.Helper()
	db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{})
	assert.NoError(t, err)
	err = db.AutoMigrate(&model.SysUser{}, &model.SysDept{})
	assert.NoError(t, err)
	return db
}

func newUser(username, nickname string, gender, status int8, deptID int64, deleted int8, now time.Time) model.SysUser {
	return model.SysUser{
		BaseModel: model.BaseModel{CreatedAt: now, UpdatedAt: now},
		Username:  username,
		Nickname:  nickname,
		Gender:    gender,
		DeptID:    deptID,
		Status:    status,
		Deleted:   deleted,
	}
}

func TestUserExportHandler_GetModule(t *testing.T) {
	handler := NewUserExportHandler(nil)
	assert.Equal(t, "user", handler.GetModule())
}

func TestUserExportHandler_GetFieldConfigs(t *testing.T) {
	handler := NewUserExportHandler(nil)
	fields := handler.GetFieldConfigs()
	assert.Len(t, fields, 8)
	assert.Equal(t, "username", fields[0].Field)
	assert.Equal(t, "用户名", fields[0].Label)
	assert.Equal(t, 1, fields[0].Order)
	assert.Equal(t, "createTime", fields[7].Field)
	assert.Equal(t, "2006-01-02 15:04:05", fields[7].DateFormat)
}

func TestUserExportHandler_EstimateCount(t *testing.T) {
	db := newTestDB(t)
	now := time.Now()
	users := []model.SysUser{
		newUser("u1", "n1", 1, 1, 0, 0, now),
		newUser("u2", "n2", 1, 1, 0, 0, now),
		newUser("u3", "n3", 1, 1, 0, 1, now),
	}
	for i := range users {
		assert.NoError(t, db.Create(&users[i]).Error)
	}

	handler := NewUserExportHandler(db)
	count := handler.EstimateCount(map[string]interface{}{})
	assert.Equal(t, int64(2), count)
}

func TestUserExportHandler_EstimateCount_WithKeywords(t *testing.T) {
	db := newTestDB(t)
	now := time.Now()
	users := []model.SysUser{
		newUser("zhangsan", "张三", 1, 1, 0, 0, now),
		newUser("lisi", "李四", 1, 1, 0, 0, now),
	}
	for i := range users {
		users[i].Mobile = "13800138000"
		if i == 1 {
			users[i].Mobile = "13900139000"
		}
		assert.NoError(t, db.Create(&users[i]).Error)
	}

	handler := NewUserExportHandler(db)
	count := handler.EstimateCount(map[string]interface{}{"keywords": "zhang"})
	assert.Equal(t, int64(1), count)
}

func TestUserExportHandler_GetDataProvider_MapsRows(t *testing.T) {
	db := newTestDB(t)
	now := time.Now()
	dept1 := model.SysDept{Name: "研发部", Deleted: 0, BaseModel: model.BaseModel{CreatedAt: now, UpdatedAt: now}}
	assert.NoError(t, db.Create(&dept1).Error)

	user := newUser("zhangsan", "张三", 1, 1, dept1.ID, 0, now)
	user.Mobile = "13800138000"
	user.Email = "zhangsan@example.com"
	assert.NoError(t, db.Create(&user).Error)

	handler := NewUserExportHandler(db)
	ctx := &import_export.ExportContext{QueryParams: map[string]interface{}{}}
	provider := handler.GetDataProvider(ctx)

	rows := provider.FetchBatch(1, 1000)
	assert.Len(t, rows, 1)
	row := rows[0]
	assert.Equal(t, "zhangsan", row[0])
	assert.Equal(t, "张三", row[1])
	assert.Equal(t, "男", row[2])
	assert.Equal(t, "研发部", row[3])
	assert.Equal(t, "13800138000", row[4])
	assert.Equal(t, "zhangsan@example.com", row[5])
	assert.Equal(t, "启用", row[6])
	createTimeStr, ok := row[7].(string)
	assert.True(t, ok)
	assert.NotEmpty(t, createTimeStr)
}

func TestUserExportHandler_GetDataProvider_StatusDisabled(t *testing.T) {
	db := newTestDB(t)
	now := time.Now()
	user := newUser("u1", "n1", 2, 0, 0, 0, now)
	assert.NoError(t, db.Create(&user).Error)

	handler := NewUserExportHandler(db)
	ctx := &import_export.ExportContext{QueryParams: map[string]interface{}{}}
	provider := handler.GetDataProvider(ctx)

	rows := provider.FetchBatch(1, 1000)
	assert.Len(t, rows, 1)
	assert.Equal(t, "女", rows[0][2])
	assert.Equal(t, "禁用", rows[0][6])
}

func TestUserExportHandler_GetDataProvider_FiltersByStatus(t *testing.T) {
	db := newTestDB(t)
	now := time.Now()
	users := []model.SysUser{
		newUser("u1", "n1", 1, 1, 0, 0, now),
		newUser("u2", "n2", 1, 0, 0, 0, now),
	}
	for i := range users {
		assert.NoError(t, db.Create(&users[i]).Error)
	}

	handler := NewUserExportHandler(db)
	statusVal := 1
	ctx := &import_export.ExportContext{
		QueryParams: map[string]interface{}{"status": statusVal},
	}
	provider := handler.GetDataProvider(ctx)
	rows := provider.FetchBatch(1, 1000)
	assert.Len(t, rows, 1)
	assert.Equal(t, "u1", rows[0][0])
}

func TestUserExportHandler_GetDataProvider_FiltersByDeptId(t *testing.T) {
	db := newTestDB(t)
	now := time.Now()
	dept1 := model.SysDept{Name: "研发部", Deleted: 0, BaseModel: model.BaseModel{CreatedAt: now, UpdatedAt: now}}
	dept2 := model.SysDept{Name: "测试部", Deleted: 0, BaseModel: model.BaseModel{CreatedAt: now, UpdatedAt: now}}
	assert.NoError(t, db.Create(&dept1).Error)
	assert.NoError(t, db.Create(&dept2).Error)

	users := []model.SysUser{
		newUser("u1", "n1", 1, 1, dept1.ID, 0, now),
		newUser("u2", "n2", 1, 1, dept2.ID, 0, now),
	}
	for i := range users {
		assert.NoError(t, db.Create(&users[i]).Error)
	}

	handler := NewUserExportHandler(db)
	ctx := &import_export.ExportContext{
		QueryParams: map[string]interface{}{"deptId": dept1.ID},
	}
	provider := handler.GetDataProvider(ctx)
	rows := provider.FetchBatch(1, 1000)
	assert.Len(t, rows, 1)
	assert.Equal(t, "u1", rows[0][0])
	assert.Equal(t, "研发部", rows[0][3])
}

func TestUserExportHandler_GetDataProvider_EmptyPage(t *testing.T) {
	db := newTestDB(t)
	handler := NewUserExportHandler(db)
	ctx := &import_export.ExportContext{QueryParams: map[string]interface{}{}}
	provider := handler.GetDataProvider(ctx)

	rows := provider.FetchBatch(1, 1000)
	assert.Empty(t, rows)
}

func TestUserImportHandler_GetModule(t *testing.T) {
	handler := NewUserImportHandler(nil, nil, "testPassword123")
	assert.Equal(t, "user", handler.GetModule())
}

func TestUserImportHandler_GetFieldConfigs(t *testing.T) {
	handler := NewUserImportHandler(nil, nil, "testPassword123")
	fields := handler.GetFieldConfigs()
	assert.Len(t, fields, 7)
	assert.Equal(t, "username", fields[0].Field)
	assert.True(t, fields[0].Required)
	assert.Equal(t, "nickname", fields[1].Field)
	assert.True(t, fields[1].Required)
}

func TestUserImportHandler_GetTemplateSampleData(t *testing.T) {
	handler := NewUserImportHandler(nil, nil, "testPassword123")
	samples := handler.GetTemplateSampleData()
	assert.Len(t, samples, 2)
	assert.Equal(t, "user001", samples[0]["username"])
	assert.Equal(t, "测试用户1", samples[0]["nickname"])
}

func TestUserImportHandler_ImportBatch_Success(t *testing.T) {
	db := newTestDB(t)
	deptRepo := &mockDeptRepo{namesToIDs: map[string]int64{"研发部": 100}}
	handler := NewUserImportHandler(db, deptRepo, "testPassword123")

	rows := []map[string]interface{}{
		{"username": "u1", "nickname": "n1", "gender": "男", "deptName": "研发部", "mobile": "13800138000", "email": "u1@example.com", "status": "启用"},
		{"username": "u2", "nickname": "n2", "gender": "女", "deptName": "", "mobile": "13900139000", "email": "", "status": "禁用"},
	}
	result := handler.ImportBatch(rows, import_export.ImportOptions{Mode: "all"}, import_export.NoopProgressCallback{})

	assert.Equal(t, 2, result.TotalRows)
	assert.Equal(t, 2, result.SuccessCount)
	assert.Equal(t, 0, result.FailureCount)
	assert.Empty(t, result.Errors)
	assert.Equal(t, 1, deptRepo.called)
	assert.ElementsMatch(t, []string{"研发部"}, deptRepo.lastNames)

	var saved []model.SysUser
	db.Order("username ASC").Find(&saved)
	assert.Len(t, saved, 2)
	assert.Equal(t, int8(1), saved[0].Gender)
	assert.Equal(t, "13800138000", saved[0].Mobile)
	assert.Equal(t, "u1@example.com", saved[0].Email)
	assert.Equal(t, int64(100), saved[0].DeptID)
	assert.Equal(t, int8(1), saved[0].Status)

	assert.Equal(t, int8(2), saved[1].Gender)
	assert.Equal(t, int8(0), saved[1].Status)
	assert.Equal(t, int64(0), saved[1].DeptID)
}

func TestUserImportHandler_ImportBatch_MissingRequiredFields(t *testing.T) {
	db := newTestDB(t)
	deptRepo := &mockDeptRepo{}
	handler := NewUserImportHandler(db, deptRepo, "testPassword123")

	rows := []map[string]interface{}{
		{"username": "", "nickname": "n1"},
		{"username": "u2", "nickname": ""},
	}
	result := handler.ImportBatch(rows, import_export.ImportOptions{Mode: "all"}, import_export.NoopProgressCallback{})

	assert.Equal(t, 2, result.TotalRows)
	assert.Equal(t, 0, result.SuccessCount)
	assert.Equal(t, 2, result.FailureCount)
	assert.Len(t, result.Errors, 2)
	assert.Equal(t, 2, result.Errors[0].Row)
	assert.Equal(t, "username", result.Errors[0].Field)
	assert.Equal(t, 3, result.Errors[1].Row)
	assert.Equal(t, "username", result.Errors[1].Field)

	var count int64
	db.Model(&model.SysUser{}).Count(&count)
	assert.Equal(t, int64(0), count)
}

func TestUserImportHandler_ImportBatch_DuplicateUsername(t *testing.T) {
	db := newTestDB(t)
	now := time.Now()
	existing := newUser("dup", "existing", 1, 1, 0, 0, now)
	assert.NoError(t, db.Create(&existing).Error)

	deptRepo := &mockDeptRepo{}
	handler := NewUserImportHandler(db, deptRepo, "testPassword123")

	rows := []map[string]interface{}{
		{"username": "dup", "nickname": "n1"},
		{"username": "new", "nickname": "n2"},
	}
	result := handler.ImportBatch(rows, import_export.ImportOptions{Mode: "partial"}, import_export.NoopProgressCallback{})

	assert.Equal(t, 2, result.TotalRows)
	assert.Equal(t, 1, result.SuccessCount)
	assert.Equal(t, 1, result.FailureCount)
	assert.Len(t, result.Errors, 1)
	assert.Equal(t, 2, result.Errors[0].Row)
	assert.Equal(t, "username", result.Errors[0].Field)
	assert.Contains(t, result.Errors[0].Message, "已存在")

	var count int64
	db.Model(&model.SysUser{}).Where("username = ?", "new").Count(&count)
	assert.Equal(t, int64(1), count)
}

func TestUserImportHandler_ImportBatch_StatusParsing(t *testing.T) {
	db := newTestDB(t)
	deptRepo := &mockDeptRepo{}
	handler := NewUserImportHandler(db, deptRepo, "testPassword123")

	rows := []map[string]interface{}{
		{"username": "u1", "nickname": "n1", "status": "0"},
		{"username": "u2", "nickname": "n2", "status": "禁用"},
		{"username": "u3", "nickname": "n3", "status": "启用"},
	}
	result := handler.ImportBatch(rows, import_export.ImportOptions{Mode: "all"}, import_export.NoopProgressCallback{})

	assert.Equal(t, 3, result.SuccessCount)

	var u1, u2, u3 model.SysUser
	db.Where("username = ?", "u1").First(&u1)
	db.Where("username = ?", "u2").First(&u2)
	db.Where("username = ?", "u3").First(&u3)
	assert.Equal(t, int8(0), u1.Status)
	assert.Equal(t, int8(0), u2.Status)
	assert.Equal(t, int8(1), u3.Status)
}

func TestUserImportHandler_ImportBatch_GenderParsing(t *testing.T) {
	db := newTestDB(t)
	deptRepo := &mockDeptRepo{}
	handler := NewUserImportHandler(db, deptRepo, "testPassword123")

	rows := []map[string]interface{}{
		{"username": "u1", "nickname": "n1", "gender": "男"},
		{"username": "u2", "nickname": "n2", "gender": "女"},
		{"username": "u3", "nickname": "n3", "gender": ""},
		{"username": "u4", "nickname": "n4", "gender": "未知"},
	}
	result := handler.ImportBatch(rows, import_export.ImportOptions{Mode: "all"}, import_export.NoopProgressCallback{})

	assert.Equal(t, 3, result.SuccessCount)
	assert.Equal(t, 1, result.FailureCount)
	assert.Contains(t, result.Errors[0].Message, "性别取值无效")

	var u1, u2, u3, u4 model.SysUser
	db.Where("username = ?", "u1").First(&u1)
	db.Where("username = ?", "u2").First(&u2)
	db.Where("username = ?", "u3").First(&u3)
	db.Where("username = ?", "u4").First(&u4)
	assert.Equal(t, int8(1), u1.Gender)
	assert.Equal(t, int8(2), u2.Gender)
	assert.Equal(t, int8(1), u3.Gender)
	assert.Equal(t, int8(0), u4.Gender)
}

func TestUserImportHandler_ImportBatch_DeptNameResolution(t *testing.T) {
	db := newTestDB(t)
	deptRepo := &mockDeptRepo{namesToIDs: map[string]int64{"研发部": 100, "测试部": 200}}
	handler := NewUserImportHandler(db, deptRepo, "testPassword123")

	rows := []map[string]interface{}{
		{"username": "u1", "nickname": "n1", "deptName": "研发部"},
		{"username": "u2", "nickname": "n2", "deptName": "测试部"},
		{"username": "u3", "nickname": "n3", "deptName": "未知部门"},
	}
	result := handler.ImportBatch(rows, import_export.ImportOptions{Mode: "all"}, import_export.NoopProgressCallback{})

	assert.Equal(t, 3, result.SuccessCount)

	var u1, u2, u3 model.SysUser
	db.Where("username = ?", "u1").First(&u1)
	db.Where("username = ?", "u2").First(&u2)
	db.Where("username = ?", "u3").First(&u3)
	assert.Equal(t, int64(100), u1.DeptID)
	assert.Equal(t, int64(200), u2.DeptID)
	assert.Equal(t, int64(0), u3.DeptID)
}

func TestUserImportHandler_ImportBatch_CallbackCancelled(t *testing.T) {
	db := newTestDB(t)
	deptRepo := &mockDeptRepo{}
	handler := NewUserImportHandler(db, deptRepo, "testPassword123")

	rows := []map[string]interface{}{
		{"username": "u1", "nickname": "n1"},
		{"username": "u2", "nickname": "n2"},
	}
	cancelledCallback := import_export.ProgressCallbackFunc{
		IsCancelledFn: func() bool { return true },
	}
	result := handler.ImportBatch(rows, import_export.ImportOptions{Mode: "all"}, cancelledCallback)

	assert.Equal(t, 2, result.TotalRows)
	assert.Equal(t, 0, result.SuccessCount)

	var count int64
	db.Model(&model.SysUser{}).Count(&count)
	assert.Equal(t, int64(0), count)
}

func TestUserImportHandler_ImportBatch_DeptRepoError_FallsBackToZero(t *testing.T) {
	db := newTestDB(t)
	deptRepo := &mockDeptRepo{err: assert.AnError}
	handler := NewUserImportHandler(db, deptRepo, "testPassword123")

	rows := []map[string]interface{}{
		{"username": "u1", "nickname": "n1", "deptName": "研发部"},
	}
	result := handler.ImportBatch(rows, import_export.ImportOptions{Mode: "all"}, import_export.NoopProgressCallback{})

	assert.Equal(t, 1, result.SuccessCount)
	var u1 model.SysUser
	db.Where("username = ?", "u1").First(&u1)
	assert.Equal(t, int64(0), u1.DeptID)
}

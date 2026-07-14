package database

import (
	"context"
	"reflect"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"gorm.io/gorm"
)

// UserClaims 用户声明接口
// 用于从上下文中提取用户ID信息
type UserClaims interface {
	GetUserID() int64
}

// DataScopeClaims 数据权限声明接口
// 用于在中间件中提取用户的数据权限范围和部门ID，注入到 context.Context
// DataScopePlugin 通过 db.Statement.Context 读取，实现异步上下文友好的行级过滤
type DataScopeClaims interface {
	UserClaims
	GetDeptID() int64
	GetDataScope() int8
}

// userContextKey 上下文键，用于存储用户ID
type userContextKey struct{}

// dataScopeContextKey 上下文键，用于存储数据权限范围
type dataScopeContextKey struct{}

// deptIDContextKey 上下文键，用于存储部门ID
type deptIDContextKey struct{}

// SetUserID 设置当前请求的用户ID到上下文
// 通常在认证中间件中调用
func SetUserID(ctx context.Context, userID int64) context.Context {
	return context.WithValue(ctx, userContextKey{}, userID)
}

// GetUserID 从上下文中获取当前请求的用户ID
// 如果上下文中没有用户ID，返回0
func GetUserID(ctx context.Context) int64 {
	if userID, ok := ctx.Value(userContextKey{}).(int64); ok {
		return userID
	}
	return 0
}

// SetDataScope 设置数据权限范围到上下文
func SetDataScope(ctx context.Context, dataScope int8) context.Context {
	return context.WithValue(ctx, dataScopeContextKey{}, dataScope)
}

// GetDataScope 从上下文获取数据权限范围
// 未设置时返回 DataScopeAll（全部数据，不过滤），保证异步任务无上下文时不误过滤
func GetDataScope(ctx context.Context) int8 {
	if dataScope, ok := ctx.Value(dataScopeContextKey{}).(int8); ok {
		return dataScope
	}
	return DataScopeAll
}

// SetDeptID 设置部门ID到上下文
func SetDeptID(ctx context.Context, deptID int64) context.Context {
	return context.WithValue(ctx, deptIDContextKey{}, deptID)
}

// GetDeptID 从上下文获取部门ID
func GetDeptID(ctx context.Context) int64 {
	if deptID, ok := ctx.Value(deptIDContextKey{}).(int64); ok {
		return deptID
	}
	return 0
}

// RegisterGormCallbacks 注册GORM回调
// 在初始化数据库后调用，用于自动填充create_by和update_by字段
func RegisterGormCallbacks(db *gorm.DB) *gorm.DB {
	if db == nil {
		return db
	}

	// 注册创建前回调
	db.Callback().Create().Before("gorm:create").Register("auto_fill_create_by", autoFillCreateBy)

	// 注册更新前回调
	db.Callback().Update().Before("gorm:update").Register("auto_fill_update_by", autoFillUpdateBy)

	return db
}

// autoFillCreateBy 创建前自动填充create_by和update_by
// 支持单个结构体和批量切片两种场景
func autoFillCreateBy(db *gorm.DB) {
	userID := GetUserID(db.Statement.Context)
	if userID == 0 {
		userID = common.SystemUserID
	}

	dest := db.Statement.Dest
	if dest == nil {
		return
	}

	val := reflect.ValueOf(dest)
	if val.Kind() == reflect.Ptr {
		val = val.Elem()
	}

	switch val.Kind() {
	case reflect.Struct:
		setAuditField(val.FieldByName("CreateBy"), userID)
		setAuditField(val.FieldByName("UpdateBy"), userID)
	case reflect.Slice, reflect.Array:
		for i := 0; i < val.Len(); i++ {
			elem := val.Index(i)
			if elem.Kind() == reflect.Ptr {
				elem = elem.Elem()
			}
			if elem.Kind() == reflect.Struct {
				setAuditField(elem.FieldByName("CreateBy"), userID)
				setAuditField(elem.FieldByName("UpdateBy"), userID)
			}
		}
	}
}

// autoFillUpdateBy 更新前自动填充update_by
// 支持单个结构体、Map和批量切片三种场景
func autoFillUpdateBy(db *gorm.DB) {
	userID := GetUserID(db.Statement.Context)
	if userID == 0 {
		userID = common.SystemUserID
	}

	dest := db.Statement.Dest
	if dest == nil {
		return
	}

	// 先检查是否为Map类型（用于Updates(map)方式）
	destVal := reflect.ValueOf(dest)
	if destVal.Kind() == reflect.Map {
		db.Statement.SetColumn("update_by", userID)
		return
	}

	// 处理结构体类型
	val := destVal
	if val.Kind() == reflect.Ptr {
		val = val.Elem()
	}

	switch val.Kind() {
	case reflect.Struct:
		setAuditField(val.FieldByName("UpdateBy"), userID)
	case reflect.Slice, reflect.Array:
		for i := 0; i < val.Len(); i++ {
			elem := val.Index(i)
			if elem.Kind() == reflect.Ptr {
				elem = elem.Elem()
			}
			if elem.Kind() == reflect.Struct {
				setAuditField(elem.FieldByName("UpdateBy"), userID)
			}
		}
	}
}

// setAuditField 设置审计字段，兼容 int64 和 *int64 两种类型
func setAuditField(field reflect.Value, userID int64) {
	if !field.IsValid() || !field.CanSet() {
		return
	}
	switch field.Kind() {
	case reflect.Int64:
		field.SetInt(userID)
	case reflect.Ptr:
		if field.Type().Elem().Kind() == reflect.Int64 {
			field.Set(reflect.ValueOf(&userID))
		}
	}
}

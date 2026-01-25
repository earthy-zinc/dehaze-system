package database

import (
	"context"
	"reflect"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
)

// BaseModel 基础模型结构体
// 包含所有表的通用字段，其他模型通过内嵌此结构体继承这些字段
type BaseModel struct {
	CreateTime time.Time `gorm:"column:create_time;type:datetime;autoCreateTime;comment:创建时间" json:"createTime"`
	UpdateTime time.Time `gorm:"column:update_time;type:datetime;autoUpdateTime;comment:更新时间" json:"updateTime"`
	CreateBy   int64     `gorm:"column:create_by;type:bigint;comment:创建人ID" json:"createBy"`
	UpdateBy   int64     `gorm:"column:update_by;type:bigint;comment:修改人ID" json:"updateBy"`
}

// UserClaims 用户声明接口
// 用于从上下文中提取用户ID信息
type UserClaims interface {
	GetUserID() int64
}

// userContextKey 上下文键，用于存储用户ID
type userContextKey struct{}

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

// 全局Gin上下文存储（线程安全）
var currentGinContext atomicGinContext

// atomicGinContext 线程安全的Gin上下文存储
type atomicGinContext struct {
	mu  sync.RWMutex
	ctx *gin.Context
}

// Set 设置Gin上下文（线程安全）
func (a *atomicGinContext) Set(c *gin.Context) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.ctx = c
}

// Get 获取Gin上下文（线程安全）
func (a *atomicGinContext) Get() *gin.Context {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.ctx
}

// Clear 清除Gin上下文（线程安全）
func (a *atomicGinContext) Clear() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.ctx = nil
}

// SetCurrentGinContext 设置当前请求的Gin上下文
func SetCurrentGinContext(c *gin.Context) {
	currentGinContext.Set(c)
}

// GetCurrentGinContext 获取当前请求的Gin上下文
func GetCurrentGinContext() *gin.Context {
	return currentGinContext.Get()
}

// ClearCurrentGinContext 清除当前请求的Gin上下文
func ClearCurrentGinContext() {
	currentGinContext.Clear()
}

// GormContextMiddleware GORM上下文中间件
// 在每个请求周期开始时设置Gin上下文，结束时清除
func GormContextMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		SetCurrentGinContext(c)
		defer ClearCurrentGinContext()
		c.Next()
	}
}

// RegisterGormCallbacks 注册GORM回调
// 在初始化数据库后调用，用于自动填充create_by和update_by字段
func RegisterGormCallbacks(db *gorm.DB) {
	if db == nil {
		return
	}

	// 注册创建前回调
	db.Callback().Create().Before("gorm:create").Register("auto_fill_create_by", autoFillCreateBy)

	// 注册更新前回调
	db.Callback().Update().Before("gorm:update").Register("auto_fill_update_by", autoFillUpdateBy)
}

// 模型字段缓存，避免重复反射提升性能
var modelFieldCache = &sync.Map{}

// fieldInfo 字段信息缓存
type fieldInfo struct {
	createByField reflect.StructField
	hasCreateBy   bool
	updateByField reflect.StructField
	hasUpdateBy   bool
}

// getCachedFieldInfo 获取缓存的字段信息
func getCachedFieldInfo(modelType reflect.Type) *fieldInfo {
	typeName := modelType.String()

	// 先从缓存获取
	if val, ok := modelFieldCache.Load(typeName); ok {
		if info, ok := val.(*fieldInfo); ok {
			return info
		}
	}

	// 缓存未命中，进行反射并缓存
	info := &fieldInfo{}
	if modelType.Kind() == reflect.Ptr {
		modelType = modelType.Elem()
	}

	if modelType.Kind() == reflect.Struct {
		createByField, ok := modelType.FieldByName("CreateBy")
		if ok && createByField.Type.Kind() == reflect.Int64 {
			info.createByField = createByField
			info.hasCreateBy = true
		}

		updateByField, ok := modelType.FieldByName("UpdateBy")
		if ok && updateByField.Type.Kind() == reflect.Int64 {
			info.updateByField = updateByField
			info.hasUpdateBy = true
		}
	}

	// 存入缓存
	modelFieldCache.Store(typeName, info)
	return info
}

// autoFillCreateBy 创建前自动填充create_by和update_by
func autoFillCreateBy(db *gorm.DB) {
	c := GetCurrentGinContext()
	if c == nil {
		return
	}

	userID := getUserIDFromGinContext(c)
	if userID == 0 {
		return
	}

	dest := db.Statement.Dest
	if dest == nil {
		return
	}

	val := reflect.ValueOf(dest)
	if val.Kind() == reflect.Ptr {
		val = val.Elem()
	}

	if val.Kind() != reflect.Struct {
		return
	}

	// 设置CreateBy字段
	if createByField := val.FieldByName("CreateBy"); createByField.IsValid() && createByField.CanSet() {
		if createByField.Kind() == reflect.Int64 {
			createByField.SetInt(userID)
		}
	}

	// 设置UpdateBy字段（创建时也设置update_by）
	if updateByField := val.FieldByName("UpdateBy"); updateByField.IsValid() && updateByField.CanSet() {
		if updateByField.Kind() == reflect.Int64 {
			updateByField.SetInt(userID)
		}
	}
}

// autoFillUpdateBy 更新前自动填充update_by
func autoFillUpdateBy(db *gorm.DB) {
	c := GetCurrentGinContext()
	if c == nil {
		return
	}

	userID := getUserIDFromGinContext(c)
	if userID == 0 {
		return
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

	if val.Kind() != reflect.Struct {
		return
	}

	// 设置UpdateBy字段
	if updateByField := val.FieldByName("UpdateBy"); updateByField.IsValid() && updateByField.CanSet() {
		if updateByField.Kind() == reflect.Int64 {
			updateByField.SetInt(userID)
		}
	}
}

// getUserIDFromGinContext 从Gin上下文中获取用户ID
func getUserIDFromGinContext(c *gin.Context) int64 {
	// 尝试从claims中获取（标准JWT中间件方式）
	if claims, exists := c.Get("claims"); exists {
		if userClaims, ok := claims.(UserClaims); ok {
			return userClaims.GetUserID()
		}

		// 尝试通过反射获取UserId字段
		val := reflect.ValueOf(claims)
		if val.Kind() == reflect.Ptr {
			val = val.Elem()
		}
		if val.Kind() == reflect.Struct {
			userIdField := val.FieldByName("UserId")
			if userIdField.IsValid() && userIdField.Kind() == reflect.Int64 {
				return userIdField.Int()
			}
			userIdField = val.FieldByName("ID")
			if userIdField.IsValid() && userIdField.Kind() == reflect.Int64 {
				return userIdField.Int()
			}
		}
	}

	// 尝试直接从上下文中获取
	if userID, exists := c.Get("userId"); exists {
		if id, ok := userID.(int64); ok {
			return id
		}
		if id, ok := userID.(int); ok {
			return int64(id)
		}
	}

	return 0
}

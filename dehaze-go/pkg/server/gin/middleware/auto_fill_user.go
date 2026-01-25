package middleware

import (
	"reflect"

	"github.com/earthyzinc/dehaze-go/pkg/database"
	"github.com/gin-gonic/gin"
)

// AutoFillUserMiddleware 自动填充用户ID中间件
// 从JWT claims中提取用户ID并设置到上下文中
// 配合GORM回调实现自动填充create_by和update_by字段
//
// 使用示例：
//
//	router.Use(common.GormContextMiddleware())
//	router.Use(middleware.JWTAuth())
//	router.Use(middleware.AutoFillUserMiddleware())
func AutoFillUserMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 从JWT claims中获取用户信息
		if claims, exists := c.Get("claims"); exists {
			// 设置用户ID到上下文，供GORM回调使用
			c.Set("userId", getUserIDFromClaims(claims))
		}

		c.Next()
	}
}

// getUserIDFromClaims 从claims中提取用户ID
func getUserIDFromClaims(claims interface{}) int64 {
	if claims == nil {
		return 0
	}

	// 尝试将claims转换为UserClaims接口
	if userClaims, ok := claims.(database.UserClaims); ok {
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
			return int64(userIdField.Int())
		}
		userIdField = val.FieldByName("ID")
		if userIdField.IsValid() && userIdField.Kind() == reflect.Int64 {
			return int64(userIdField.Int())
		}
	}

	return 0
}

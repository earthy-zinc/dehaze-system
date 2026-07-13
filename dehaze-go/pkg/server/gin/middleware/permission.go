package middleware

import (
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
)

// Permission 权限校验中间件（不支持通配符）
// 参数：
//   - perms: 需要的权限标识列表（支持多权限，用户只需满足任一）
//
// 用法示例：
//
//	router.GET("/user", middleware.Permission("user:read"), userHandler)
//	router.POST("/user", middleware.Permission("user:create"), userHandler)
//	router.POST("/user", middleware.Permission("user:create", "user:admin"), userHandler) // 多权限，满足任一即可
func Permission(perms ...string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if len(perms) == 0 {
			c.Next()
			return
		}

		// 超级管理员放行（与 Java 后端 SecurityUtils.isRoot() 逻辑一致）
		if security.IsRoot(c) {
			c.Next()
			return
		}

		// 检查是否有任一权限
		hasPerm, err := security.HasAnyPermission(c, perms...)
		if err != nil {
			_ = c.Error(common.WrapBizError(common.AUTHORIZED_ERROR, "权限校验失败", err))
			c.Abort()
			return
		}

		if !hasPerm {
			_ = c.Error(common.NewBizError(common.AUTHORIZED_ERROR, "权限不足"))
			c.Abort()
			return
		}

		c.Next()
	}
}

// PermissionWithWildcard 支持通配符的权限校验中间件
// 支持的通配符：
//   - *: 匹配任意多个字符（包括空字符）
//   - ?: 匹配单个字符
//
// 参数：
//   - perms: 需要的权限标识列表（支持多权限，用户只需满足任一）
//
// 用法示例：
//
//	router.GET("/user", middleware.PermissionWithWildcard("user:*"), userHandler) // 匹配所有user开头的权限
//	router.GET("/admin", middleware.PermissionWithWildcard("admin:read:*"), userHandler) // 匹配admin:read:xxx
//	router.GET("/data", middleware.PermissionWithWildcard("data:?"), userHandler) // 匹配data:后接单个字符的权限
func PermissionWithWildcard(perms ...string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if len(perms) == 0 {
			c.Next()
			return
		}

		// 获取用户信息
		claims := security.GetUserInfo(c)
		if claims == nil {
			_ = c.Error(common.NewBizError(common.ACCESS_UNAUTHORIZED, "未登录或非法访问，请登录"))
			c.Abort()
			return
		}

		// 检查是否有任一权限（支持通配符）
		for _, perm := range perms {
			hasPerm, err := security.HasPermissionWithWildcardList(perm, claims.Authorities)
			if err != nil {
				_ = c.Error(common.WrapBizError(common.AUTHORIZED_ERROR, "权限校验失败", err))
				c.Abort()
				return
			}
			if hasPerm {
				c.Next()
				return
			}
		}

		_ = c.Error(common.NewBizError(common.AUTHORIZED_ERROR, "权限不足"))
		c.Abort()
	}
}

// RequireAllPermission 需要满足所有权限的中间件
// 参数：
//   - perms: 需要的权限标识列表（用户需满足所有权限）
//
// 用法示例：
//
//	router.POST("/admin", middleware.RequireAllPermission("admin:create", "admin:approve"), adminHandler)
func RequireAllPermission(perms ...string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if len(perms) == 0 {
			c.Next()
			return
		}

		// 超级管理员放行
		if security.IsRoot(c) {
			c.Next()
			return
		}

		// 检查是否有所有权限
		hasAll, err := security.HasAllPermissions(c, perms...)
		if err != nil {
			_ = c.Error(common.WrapBizError(common.AUTHORIZED_ERROR, "权限校验失败", err))
			c.Abort()
			return
		}

		if !hasAll {
			_ = c.Error(common.NewBizError(common.AUTHORIZED_ERROR, "权限不足，需要满足所有权限"))
			c.Abort()
			return
		}

		c.Next()
	}
}

// RequireAllPermissionWithWildcard 需要满足所有权限的中间件（支持通配符）
// 参数：
//   - perms: 需要的权限标识列表（用户需满足所有权限，支持通配符）
//
// 用法示例：
//
//	router.POST("/admin", middleware.RequireAllPermissionWithWildcard("admin:*", "super:*"), adminHandler)
func RequireAllPermissionWithWildcard(perms ...string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if len(perms) == 0 {
			c.Next()
			return
		}

		// 获取用户信息
		claims := security.GetUserInfo(c)
		if claims == nil {
			_ = c.Error(common.NewBizError(common.ACCESS_UNAUTHORIZED, "未登录或非法访问，请登录"))
			c.Abort()
			return
		}

		// 检查是否有所有权限（支持通配符）
		for _, perm := range perms {
			hasPerm, err := security.HasPermissionWithWildcardList(perm, claims.Authorities)
			if err != nil {
				_ = c.Error(common.WrapBizError(common.AUTHORIZED_ERROR, "权限校验失败", err))
				c.Abort()
				return
			}
			if !hasPerm {
				_ = c.Error(common.NewBizError(common.AUTHORIZED_ERROR, "权限不足，需要满足所有权限"))
				c.Abort()
				return
			}
		}

		c.Next()
	}
}

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

		if err := CheckPermission(c, perms...); err != nil {
			_ = c.Error(err)
			c.Abort()
			return
		}

		c.Next()
	}
}

// CheckPermission handler 内调用的权限校验（校验失败返回业务错误，由调用方决定如何响应）。
// 用于"先参数绑定后权限校验"的路由：FastAPI 先做 body 校验再执行权限装饰器，
// 非法 body + 无权限的请求在 Python 端返回 A0400；gin 中间件链先于 handler 执行，
// 若这些路由继续用 Permission 中间件会返回 A0301，与 Python 行为不一致。
func CheckPermission(c *gin.Context, perms ...string) error {
	if len(perms) == 0 {
		return nil
	}

	// 超级管理员放行（与 Java 后端 SecurityUtils.isRoot() 逻辑一致）
	if security.IsRoot(c) {
		return nil
	}

	// 与 Python 端 403 语义对齐：A0301 访问未授权
	hasPerm, err := security.HasAnyPermission(c, perms...)
	if err != nil || !hasPerm {
		return common.NewBizError(common.ACCESS_UNAUTHORIZED, "访问未授权")
	}
	return nil
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

// RequireRoot 要求当前用户为超级管理员（ROOT 角色）
func RequireRoot() gin.HandlerFunc {
	return func(c *gin.Context) {
		if !security.IsRoot(c) {
			_ = c.Error(common.NewBizError(common.AUTHORIZED_ERROR, "需要超级管理员权限"))
			c.Abort()
			return
		}
		c.Next()
	}
}

// RequireAdmin 要求当前用户为管理员（ADMIN 或 ROOT 角色）
func RequireAdmin() gin.HandlerFunc {
	return func(c *gin.Context) {
		if !security.IsAdmin(c) {
			_ = c.Error(common.NewBizError(common.AUTHORIZED_ERROR, "需要管理员权限"))
			c.Abort()
			return
		}
		c.Next()
	}
}

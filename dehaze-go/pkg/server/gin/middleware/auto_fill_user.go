package middleware

import (
	"github.com/earthyzinc/dehaze-go/pkg/database"
	"github.com/gin-gonic/gin"
)

// UserContextMiddleware 从JWT claims中提取用户身份信息并注入到请求上下文
// 供 GORM 回调（autoFillCreateBy/autoFillUpdateBy）通过 db.Statement.Context
// 读取，实现审计字段（create_by/update_by）自动填充
// 同时注入 dataScope/deptID，供 DataScopePlugin 实现行级数据权限过滤
// 同时注入 IP/UserAgent，供 AuditLogService 记录审计日志
//
// DataScopePlugin 采用白名单模式：仅 DefaultDataScopeConfig.Tables 中显式配置的表
// 会被过滤，未配置的表（如 sys_role/sys_menu 等系统表）不受影响，
// 因此内部校验查询不会被误过滤。
func UserContextMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		ctx := c.Request.Context()
		ctx = database.SetIP(ctx, c.ClientIP())
		ctx = database.SetUserAgent(ctx, c.GetHeader("User-Agent"))
		if claims, exists := c.Get("claims"); exists {
			if u, ok := claims.(database.DataScopeClaims); ok {
				ctx = database.SetUserID(ctx, u.GetUserID())
				ctx = database.SetDeptID(ctx, u.GetDeptID())
				ctx = database.SetDataScope(ctx, u.GetDataScope())
			}
		}
		c.Request = c.Request.WithContext(ctx)
		c.Next()
	}
}

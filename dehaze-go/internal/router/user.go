package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterSysUserRoutes(rg *gin.RouterGroup, sysUserApi *api.SysUserApi) gin.IRoutes {
	sysUserRouter := rg.Group("users")
	{
		sysUserRouter.GET("page", sysUserApi.ListPagedUsers)
		sysUserRouter.GET(":userId/form", sysUserApi.GetUserForm)
		sysUserRouter.POST("", sysUserApi.SaveUser)
		sysUserRouter.PUT(":userId", sysUserApi.UpdateUser)
		sysUserRouter.DELETE(":ids", sysUserApi.DeleteUsers)
		sysUserRouter.PATCH(":userId/password", sysUserApi.UpdatePassword)
		sysUserRouter.PATCH(":userId/status", sysUserApi.UpdateUserStatus)
		sysUserRouter.GET("me", sysUserApi.GetCurrentUserInfo)
		sysUserRouter.GET("_export", sysUserApi.ListExportUsers)
		sysUserRouter.GET("template", sysUserApi.DownloadImportTemplate)
		sysUserRouter.POST("_import", sysUserApi.ImportUsers)
	}
	return sysUserRouter
}

func RegisterSysRoleRoutes(rg *gin.RouterGroup, sysRoleApi *api.SysRoleApi) gin.IRoutes {
	sysRoleRouter := rg.Group("roles")
	{
		// 读操作 - 无需额外权限
		sysRoleRouter.GET("page", sysRoleApi.GetRolePage)
		sysRoleRouter.GET("options", sysRoleApi.ListRoleOptions)
		sysRoleRouter.GET(":roleId/form", sysRoleApi.GetRoleForm)
		sysRoleRouter.GET(":roleId/menuIds", sysRoleApi.GetRoleMenuIds)

		// 写操作 - 需要权限校验 + 防重复提交
		sysRoleRouter.POST("", middleware.Permission("sys:role:add"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), sysRoleApi.AddRole)
		sysRoleRouter.PUT(":roleId", middleware.Permission("sys:role:edit"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), sysRoleApi.UpdateRole)
		sysRoleRouter.DELETE(":ids", middleware.Permission("sys:role:delete"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), sysRoleApi.DeleteRoles)
		sysRoleRouter.PUT(":roleId/status", middleware.Permission("sys:role:edit"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), sysRoleApi.UpdateRoleStatus)
		sysRoleRouter.PUT(":roleId/menus", middleware.Permission("sys:role:edit"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), sysRoleApi.AssignMenusToRole)
	}
	return sysRoleRouter
}

func RegisterSysDeptRoutes(rg *gin.RouterGroup, sysDeptApi *api.SysDeptApi) gin.IRoutes {
	sysDeptRouter := rg.Group("dept")
	{
		// 读操作 - 无需额外权限
		sysDeptRouter.GET("", sysDeptApi.ListDepartments)
		sysDeptRouter.GET("options", sysDeptApi.ListDeptOptions)
		sysDeptRouter.GET(":deptId/form", sysDeptApi.GetDeptForm)

		// 写操作 - 需要权限校验 + 防重复提交
		sysDeptRouter.POST("", middleware.Permission("sys:dept:add"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), sysDeptApi.SaveDept)
		sysDeptRouter.PUT(":deptId", middleware.Permission("sys:dept:edit"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), sysDeptApi.UpdateDept)
		sysDeptRouter.DELETE(":ids", middleware.Permission("sys:dept:delete"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), sysDeptApi.DeleteDepartments)
	}
	return sysDeptRouter
}

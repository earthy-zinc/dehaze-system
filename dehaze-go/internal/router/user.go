package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterSysUserRoutes(rg *gin.RouterGroup, sysUserApi *api.SysUserApi) gin.IRoutes {
	sysUserRouter := rg.Group("users")
	{
		// 读操作 - 无需额外权限
		sysUserRouter.GET("page", sysUserApi.ListPagedUsers)
		sysUserRouter.GET(":userId/form", sysUserApi.GetUserForm)
		sysUserRouter.GET("_export", sysUserApi.ListExportUsers)
		sysUserRouter.GET("template", sysUserApi.DownloadImportTemplate)

		// 写操作 - 需要权限校验（POST 新增操作加防重复提交，与 Java @PreventDuplicateSubmit 一致）
		sysUserRouter.POST("", middleware.Permission("sys:user:add"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), sysUserApi.SaveUser)
		sysUserRouter.PUT(":userId", middleware.Permission("sys:user:edit"), sysUserApi.UpdateUser)
		sysUserRouter.DELETE(":ids", middleware.Permission("sys:user:delete"), sysUserApi.DeleteUsers)
		sysUserRouter.PATCH(":userId/password", middleware.Permission("sys:user:edit"), sysUserApi.UpdatePassword)
		sysUserRouter.PATCH(":userId/status", middleware.Permission("sys:user:edit"), sysUserApi.UpdateUserStatus)
		sysUserRouter.POST("_import", middleware.Permission("sys:user:add"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 5}), sysUserApi.ImportUsers)
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

		// 写操作 - 需要权限校验（POST 新增操作加防重复提交，与 Java @PreventDuplicateSubmit 一致）
		sysRoleRouter.POST("", middleware.Permission("sys:role:add"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), sysRoleApi.AddRole)
		sysRoleRouter.PUT(":roleId", middleware.Permission("sys:role:edit"), sysRoleApi.UpdateRole)
		sysRoleRouter.DELETE(":ids", middleware.Permission("sys:role:delete"), sysRoleApi.DeleteRoles)
		sysRoleRouter.PUT(":roleId/status", middleware.Permission("sys:role:edit"), sysRoleApi.UpdateRoleStatus)
		sysRoleRouter.PATCH(":roleId/menus", middleware.Permission("sys:role:edit"), sysRoleApi.AssignMenusToRole)
	}
	return sysRoleRouter
}

func RegisterSysDeptRoutes(rg *gin.RouterGroup, sysDeptApi *api.SysDeptApi) gin.IRoutes {
	sysDeptRouter := rg.Group("depts")
	{
		// 读操作 - 无需额外权限
		sysDeptRouter.GET("", sysDeptApi.ListDepartments)
		sysDeptRouter.GET("options", sysDeptApi.ListDeptOptions)
		sysDeptRouter.GET(":deptId/form", sysDeptApi.GetDeptForm)

		// 写操作 - 需要权限校验（POST 新增操作加防重复提交，与 Java @PreventDuplicateSubmit 一致）
		sysDeptRouter.POST("", middleware.Permission("sys:dept:add"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), sysDeptApi.SaveDept)
		sysDeptRouter.PUT(":deptId", middleware.Permission("sys:dept:edit"), sysDeptApi.UpdateDept)
		sysDeptRouter.DELETE(":ids", middleware.Permission("sys:dept:delete"), sysDeptApi.DeleteDepartments)
	}
	return sysDeptRouter
}

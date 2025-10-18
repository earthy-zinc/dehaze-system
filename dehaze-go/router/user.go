package router

import (
	"github.com/earthyzinc/dehaze-go/api"
	"github.com/gin-gonic/gin"
)

type UserRouter struct{}

func (r *UserRouter) InitUserRouter(Router *gin.RouterGroup) (R gin.IRoutes) {
	userRouter := Router.Group("user")
	{
		userRouter.POST("login", authApi.Login)
		userRouter.POST("captcha", authApi.Captcha)
	}
	return userRouter
}

type SysUserRouter struct{}

func (r *SysUserRouter) InitSysUserRouter(Router *gin.RouterGroup) (R gin.IRoutes) {
	sysUserRouter := Router.Group("users")
	sysUserApi := api.ApiGroupApp.SysUserApi
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
	}
	return sysUserRouter
}

type SysRoleRouter struct{}

func (r *SysRoleRouter) InitSysRoleRouter(Router *gin.RouterGroup) (R gin.IRoutes) {
	sysRoleRouter := Router.Group("roles")
	sysRoleApi := api.ApiGroupApp.SysRoleApi
	{
		sysRoleRouter.GET("page", sysRoleApi.GetRolePage)
		sysRoleRouter.GET("options", sysRoleApi.ListRoleOptions)
		sysRoleRouter.POST("", sysRoleApi.AddRole)
		sysRoleRouter.GET(":roleId/form", sysRoleApi.GetRoleForm)
		sysRoleRouter.PUT(":id", sysRoleApi.UpdateRole)
		sysRoleRouter.DELETE(":ids", sysRoleApi.DeleteRoles)
		sysRoleRouter.PUT(":roleId/status", sysRoleApi.UpdateRoleStatus)
		sysRoleRouter.GET(":roleId/menuIds", sysRoleApi.GetRoleMenuIds)
		sysRoleRouter.PUT(":roleId/menus", sysRoleApi.AssignMenusToRole)
	}
	return sysRoleRouter
}

type SysDeptRouter struct{}

func (r *SysDeptRouter) InitSysDeptRouter(Router *gin.RouterGroup) (R gin.IRoutes) {
	sysDeptRouter := Router.Group("dept")
	sysDeptApi := api.ApiGroupApp.SysDeptApi
	{
		sysDeptRouter.GET("", sysDeptApi.ListDepartments)
		sysDeptRouter.GET("options", sysDeptApi.ListDeptOptions)
		sysDeptRouter.GET(":deptId/form", sysDeptApi.GetDeptForm)
		sysDeptRouter.POST("", sysDeptApi.SaveDept)
		sysDeptRouter.PUT(":deptId", sysDeptApi.UpdateDept)
		sysDeptRouter.DELETE(":ids", sysDeptApi.DeleteDepartments)
	}
	return sysDeptRouter
}
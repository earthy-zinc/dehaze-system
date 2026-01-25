package api

import (
	"strconv"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/service"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"github.com/gin-gonic/gin"
)

type SysUserApi struct {
	userServiceExtend service.UserServiceExtend
}

// ListPagedUsers 用户分页列表
// @Summary 用户分页列表
// @Description 获取用户分页列表
// @Tags 用户接口
// @Accept application/json
// @Produce application/json
// @Param keywords query string false "关键字(用户名/昵称/手机号)"
// @Param status query int false "用户状态"
// @Param deptId query int false "部门ID"
// @Param startTime query string false "创建时间-开始时间"
// @Param endTime query string false "创建时间-结束时间"
// @Param pageNum query int false "页码"
// @Param pageSize query int false "每页条数"
// @Success 200 {object} vo.Result{data=vo.PageResult[vo.UserPageVO]}
// @Router /api/v1/users/page [get]
func (api *SysUserApi) ListPagedUsers(c *gin.Context) {
	// 解析查询参数
	var queryParams query.UserPageQuery
	queryParams.Keywords = c.Query("keywords")
	if statusStr := c.Query("status"); statusStr != "" {
		if status, err := strconv.Atoi(statusStr); err == nil {
			queryParams.Status = &status
		}
	}
	if deptIdStr := c.Query("deptId"); deptIdStr != "" {
		if deptId, err := strconv.ParseInt(deptIdStr, 10, 64); err == nil {
			queryParams.DeptId = &deptId
		}
	}
	queryParams.StartTime = c.Query("startTime")
	queryParams.EndTime = c.Query("endTime")

	if pageNumStr := c.Query("pageNum"); pageNumStr != "" {
		if pageNum, err := strconv.Atoi(pageNumStr); err == nil {
			queryParams.PageNum = pageNum
		} else {
			queryParams.PageNum = 1
		}
	} else {
		queryParams.PageNum = 1
	}

	if pageSizeStr := c.Query("pageSize"); pageSizeStr != "" {
		if pageSize, err := strconv.Atoi(pageSizeStr); err == nil {
			queryParams.PageSize = pageSize
		} else {
			queryParams.PageSize = 10
		}
	} else {
		queryParams.PageSize = 10
	}

	// 调用服务获取分页数据
	result, err := api.userServiceExtend.ListPagedUsers(queryParams)
	if err != nil {
		common.FailWithMessage("获取用户分页列表失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(result, "查询成功", c)
}

// GetUserForm 获取用户表单数据
// @Summary 用户表单数据
// @Description 获取用户表单数据
// @Tags 用户接口
// @Accept application/json
// @Produce application/json
// @Param userId path int true "用户ID"
// @Success 200 {object} vo.Result{data=bo.UserFormBO}
// @Router /api/v1/users/{userId}/form [get]
func (api *SysUserApi) GetUserForm(c *gin.Context) {
	// 获取路径参数
	userIdStr := c.Param("userId")
	userId, err := strconv.ParseInt(userIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("用户ID格式不正确", c)
		return
	}

	// 调用服务获取用户表单数据
	userFormBO, err := api.userServiceExtend.GetUserFormData(userId)
	if err != nil {
		common.FailWithMessage("获取用户表单数据失败: "+err.Error(), c)
		return
	}

	// 用户不存在时返回null（与Java行为一致）
	if userFormBO.ID == 0 {
		common.OkWithDetailed(nil, "查询成功", c)
		return
	}

	common.OkWithDetailed(userFormBO, "查询成功", c)
}

// SaveUser 新增用户
// @Summary 新增用户
// @Description 新增用户
// @Tags 用户接口
// @Accept application/json
// @Produce application/json
// @Param userForm body bo.UserFormBO true "用户表单对象"
// @Success 200 {object} vo.Result
// @Router /api/v1/users [post]
func (api *SysUserApi) SaveUser(c *gin.Context) {
	// 绑定请求参数
	var userFormBO bo.UserFormBO
	if err := c.ShouldBindJSON(&userFormBO); err != nil {
		common.FailWithMessage("请求参数解析失败: "+err.Error(), c)
		return
	}

	// 调用服务保存用户
	err := api.userServiceExtend.SaveUser(userFormBO)
	if err != nil {
		common.FailWithMessage("新增用户失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("新增用户成功", c)
}

// UpdateUser 修改用户
// @Summary 修改用户
// @Description 修改用户
// @Tags 用户接口
// @Accept application/json
// @Produce application/json
// @Param userId path int true "用户ID"
// @Param userForm body bo.UserFormBO true "用户表单对象"
// @Success 200 {object} vo.Result
// @Router /api/v1/users/{userId} [put]
func (api *SysUserApi) UpdateUser(c *gin.Context) {
	// 获取路径参数
	userIdStr := c.Param("userId")
	userId, err := strconv.ParseInt(userIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("用户ID格式不正确", c)
		return
	}

	// 绑定请求参数
	var userFormBO bo.UserFormBO
	if err := c.ShouldBindJSON(&userFormBO); err != nil {
		common.FailWithMessage("请求参数解析失败: "+err.Error(), c)
		return
	}

	// 调用服务更新用户
	err = api.userServiceExtend.UpdateUser(userId, userFormBO)
	if err != nil {
		common.FailWithMessage("修改用户失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("修改用户成功", c)
}

// DeleteUsers 删除用户
// @Summary 删除用户
// @Description 删除用户
// @Tags 用户接口
// @Accept application/json
// @Produce application/json
// @Param ids path string true "用户ID，多个以英文逗号(,)分割"
// @Success 200 {object} vo.Result
// @Router /api/v1/users/{ids} [delete]
func (api *SysUserApi) DeleteUsers(c *gin.Context) {
	// 获取路径参数
	ids := c.Param("ids")

	// 调用服务删除用户
	err := api.userServiceExtend.DeleteUsers(ids)
	if err != nil {
		common.FailWithMessage("删除用户失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("删除用户成功", c)
}

// UpdatePassword 修改用户密码
// @Summary 修改用户密码
// @Description 修改用户密码
// @Tags 用户接口
// @Accept application/json
// @Produce application/json
// @Param userId path int true "用户ID"
// @Param password query string true "新密码"
// @Success 200 {object} vo.Result
// @Router /api/v1/users/{userId}/password [patch]
func (api *SysUserApi) UpdatePassword(c *gin.Context) {
	// 获取路径参数
	userIdStr := c.Param("userId")
	userId, err := strconv.ParseInt(userIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("用户ID格式不正确", c)
		return
	}

	// 获取查询参数
	password := c.Query("password")

	// 调用服务更新密码
	err = api.userServiceExtend.UpdatePassword(userId, password)
	if err != nil {
		common.FailWithMessage("修改密码失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("修改密码成功", c)
}

// UpdateUserStatus 修改用户状态
// @Summary 修改用户状态
// @Description 修改用户状态
// @Tags 用户接口
// @Accept application/json
// @Produce application/json
// @Param userId path int true "用户ID"
// @Param status query int true "用户状态(1:启用;0:禁用)"
// @Success 200 {object} vo.Result
// @Router /api/v1/users/{userId}/status [patch]
func (api *SysUserApi) UpdateUserStatus(c *gin.Context) {
	// 获取路径参数
	userIdStr := c.Param("userId")
	userId, err := strconv.ParseInt(userIdStr, 10, 64)
	if err != nil {
		common.FailWithMessage("用户ID格式不正确", c)
		return
	}

	// 获取查询参数
	statusStr := c.Query("status")
	status, err := strconv.Atoi(statusStr)
	if err != nil {
		common.FailWithMessage("状态参数格式不正确", c)
		return
	}

	// 调用服务更新用户状态
	err = api.userServiceExtend.UpdateUserStatus(userId, int8(status))
	if err != nil {
		common.FailWithMessage("修改用户状态失败: "+err.Error(), c)
		return
	}

	common.OkWithMessage("修改用户状态成功", c)
}

// GetCurrentUserInfo 获取当前登录用户信息
// @Summary 获取当前登录用户信息
// @Description 获取当前登录用户信息
// @Tags 用户接口
// @Accept application/json
// @Produce application/json
// @Success 200 {object} vo.Result{data=vo.UserInfoVO}
// @Router /api/v1/users/me [get]
func (api *SysUserApi) GetCurrentUserInfo(c *gin.Context) {
	// TODO: 这里应该从上下文中获取当前登录用户信息
	// 简化处理，假设从某个地方获取到用户名
	// 实际应该从token中解析
	username := c.GetString("username")
	if username == "" {
		username = "admin" // 默认值，实际应从上下文获取
	}

	// 调用服务获取当前用户信息
	userInfoVO, err := api.userServiceExtend.GetCurrentUserInfo(username)
	if err != nil {
		common.FailWithMessage("获取当前用户信息失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(userInfoVO, "查询成功", c)
}

// ListExportUsers 导出用户
// @Summary 导出用户
// @Description 导出用户列表
// @Tags 用户接口
// @Accept application/json
// @Produce application/json
// @Param keywords query string false "关键字(用户名/昵称/手机号)"
// @Param status query int false "用户状态"
// @Param deptId query int false "部门ID"
// @Param startTime query string false "创建时间-开始时间"
// @Param endTime query string false "创建时间-结束时间"
// @Success 200 {object} vo.Result{data=[]vo.UserExportVO}
// @Router /api/v1/users/_export [get]
func (api *SysUserApi) ListExportUsers(c *gin.Context) {
	// 解析查询参数
	var queryParams query.UserPageQuery
	queryParams.Keywords = c.Query("keywords")
	if statusStr := c.Query("status"); statusStr != "" {
		if status, err := strconv.Atoi(statusStr); err == nil {
			queryParams.Status = &status
		}
	}
	if deptIdStr := c.Query("deptId"); deptIdStr != "" {
		if deptId, err := strconv.ParseInt(deptIdStr, 10, 64); err == nil {
			queryParams.DeptId = &deptId
		}
	}
	queryParams.StartTime = c.Query("startTime")
	queryParams.EndTime = c.Query("endTime")

	// 调用服务获取导出数据
	userExportVOs, err := api.userServiceExtend.ListExportUsers(queryParams)
	if err != nil {
		common.FailWithMessage("获取导出用户列表失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(userExportVOs, "查询成功", c)
}

// DownloadImportTemplate 下载导入模板
// @Summary 下载用户导入模板
// @Description 下载用户导入模板Excel文件
// @Tags 用户接口
// @Accept application/json
// @Produce application/vnd.ms-excel
// @Success 200 {file} file
// @Router /api/v1/users/template [get]
func (api *SysUserApi) DownloadImportTemplate(c *gin.Context) {
	filePath, err := api.userServiceExtend.DownloadImportTemplate()
	if err != nil {
		common.FailWithMessage("下载导入模板失败: "+err.Error(), c)
		return
	}
	defer func() {
		if filePath != "" {
			_ = utils.DeleteTempFile(filePath)
		}
	}()

	c.Header("Content-Description", "File Transfer")
	c.Header("Content-Disposition", "attachment; filename=user_import_template.xlsx")
	c.File(filePath)
}

// ImportUsers 导入用户
// @Summary 导入用户
// @Description 通过Excel文件导入用户
// @Tags 用户接口
// @Accept multipart/form-data
// @Produce application/json
// @Param file formData file true "Excel文件"
// @Success 200 {object} vo.Result{data=vo.ImportResultVO}
// @Router /api/v1/users/_import [post]
func (api *SysUserApi) ImportUsers(c *gin.Context) {
	// 获取上传的文件
	file, err := c.FormFile("file")
	if err != nil {
		common.FailWithMessage("获取上传文件失败: "+err.Error(), c)
		return
	}

	// 打开上传的文件
	src, err := file.Open()
	if err != nil {
		common.FailWithMessage("打开上传文件失败: "+err.Error(), c)
		return
	}
	defer src.Close()

	// 调用服务导入用户
	importResult, err := api.userServiceExtend.ImportUsers(src)
	if err != nil {
		common.FailWithMessage("导入用户失败: "+err.Error(), c)
		return
	}

	common.OkWithDetailed(importResult, "导入成功", c)
}

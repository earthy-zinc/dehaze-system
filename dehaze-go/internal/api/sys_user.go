package api

import (
	"strconv"
	"strings"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	userservice "github.com/earthyzinc/dehaze-go/internal/service/user"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/earthyzinc/dehaze-go/pkg/utils"
	"github.com/gin-gonic/gin"
)

type SysUserApi struct {
	userService userservice.IUserService
}

func NewSysUserApi(userService userservice.IUserService) *SysUserApi {
	return &SysUserApi{
		userService: userService,
	}
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
// @Success 200 {object} common.Response{data=common.PageResult}
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
	result, err := api.userService.GetPage(c.Request.Context(), &queryParams)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(result, common.SUCCESS.Msg, c)
}

// GetUserForm 获取用户表单数据
// @Summary 用户表单数据
// @Description 获取用户表单数据
// @Tags 用户接口
// @Accept application/json
// @Produce application/json
// @Param userId path int true "用户ID"
// @Success 200 {object} common.Response{data=bo.UserFormBO}
// @Router /api/v1/users/{userId}/form [get]
func (api *SysUserApi) GetUserForm(c *gin.Context) {
	// 获取路径参数
	userIdStr := c.Param("userId")
	userId, err := strconv.ParseInt(userIdStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "用户ID格式不正确"))
		return
	}

	// 调用服务获取用户表单数据
	formData, err := api.userService.GetFormData(c.Request.Context(), userId)
	if err != nil {
		_ = c.Error(err)
		return
	}

	// 用户不存在时返回null（与Java行为一致）
	if formData == nil || formData.ID == 0 {
		common.OkWithDetailed(nil, common.SUCCESS.Msg, c)
		return
	}

	common.OkWithDetailed(formData, common.SUCCESS.Msg, c)
}

// SaveUser 新增用户
// @Summary 新增用户
// @Description 新增用户
// @Tags 用户接口
// @Accept application/json
// @Produce application/json
// @Param userForm body bo.UserFormBO true "用户表单对象"
// @Success 200 {object} common.Response
// @Router /api/v1/users [post]
func (api *SysUserApi) SaveUser(c *gin.Context) {
	// 绑定请求参数
	var userFormBO bo.UserFormBO
	if err := c.ShouldBindJSON(&userFormBO); err != nil {
		_ = c.Error(err)
		return
	}

	// 调用服务保存用户
	err := api.userService.Create(c.Request.Context(), &userFormBO)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response
// @Router /api/v1/users/{userId} [put]
func (api *SysUserApi) UpdateUser(c *gin.Context) {
	// 获取路径参数
	userIdStr := c.Param("userId")
	userId, err := strconv.ParseInt(userIdStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "用户ID格式不正确"))
		return
	}

	// 绑定请求参数
	var userFormBO bo.UserFormBO
	if err := c.ShouldBindJSON(&userFormBO); err != nil {
		_ = c.Error(err)
		return
	}

	// 调用服务更新用户
	err = api.userService.Update(c.Request.Context(), userId, &userFormBO)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response
// @Router /api/v1/users/{ids} [delete]
func (api *SysUserApi) DeleteUsers(c *gin.Context) {
	// 获取路径参数
	idsStr := c.Param("ids")
	if idsStr == "" {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "删除的用户数据为空"))
		return
	}

	// 解析ID列表
	idStrings := strings.Split(idsStr, ",")
	var ids []int64
	for _, idStr := range idStrings {
		id, err := strconv.ParseInt(idStr, 10, 64)
		if err != nil {
			_ = c.Error(common.NewBizError(common.PARAM_ERROR, "用户ID格式不正确"))
			return
		}
		ids = append(ids, id)
	}

	// 调用服务删除用户
	err := api.userService.Delete(c.Request.Context(), ids)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response
// @Router /api/v1/users/{userId}/password [patch]
func (api *SysUserApi) UpdatePassword(c *gin.Context) {
	// 获取路径参数
	userIdStr := c.Param("userId")
	userId, err := strconv.ParseInt(userIdStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "用户ID格式不正确"))
		return
	}

	// 获取查询参数
	password := c.Query("password")

	// 调用服务更新密码
	err = api.userService.UpdatePassword(c.Request.Context(), userId, password)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response
// @Router /api/v1/users/{userId}/status [patch]
func (api *SysUserApi) UpdateUserStatus(c *gin.Context) {
	// 获取路径参数
	userIdStr := c.Param("userId")
	userId, err := strconv.ParseInt(userIdStr, 10, 64)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "用户ID格式不正确"))
		return
	}

	// 获取查询参数
	statusStr := c.Query("status")
	status, err := strconv.Atoi(statusStr)
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "状态参数格式不正确"))
		return
	}

	// 调用服务更新用户状态
	err = api.userService.UpdateStatus(c.Request.Context(), userId, int8(status))
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response{data=vo.UserInfoVO}
// @Router /api/v1/users/me [get]
func (api *SysUserApi) GetCurrentUserInfo(c *gin.Context) {
	userID := security.GetUserID(c)

	// 调用服务获取当前用户信息
	userInfoVO, err := api.userService.GetCurrentUserInfo(c.Request.Context(), userID)
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response{data=[]vo.UserExportVO}
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
	userExportVOs, err := api.userService.ExportUsers(c.Request.Context(), &queryParams)
	if err != nil {
		_ = c.Error(err)
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
	filePath, err := api.userService.DownloadImportTemplate(c.Request.Context())
	if err != nil {
		_ = c.Error(err)
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
// @Success 200 {object} common.Response{data=vo.ImportResultVO}
// @Router /api/v1/users/_import [post]
func (api *SysUserApi) ImportUsers(c *gin.Context) {
	// 获取上传的文件
	file, err := c.FormFile("file")
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "文件上传失败"))
		return
	}

	// 打开上传的文件
	src, err := file.Open()
	if err != nil {
		_ = c.Error(common.NewBizError(common.PARAM_ERROR, "打开上传文件失败"))
		return
	}
	defer src.Close()

	// 调用服务导入用户
	importResult, err := api.userService.ImportUsersFromFile(c.Request.Context(), src)
	if err != nil {
		_ = c.Error(err)
		return
	}

	common.OkWithDetailed(importResult, "导入成功", c)
}

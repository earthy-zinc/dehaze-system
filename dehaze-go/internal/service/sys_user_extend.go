package service

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/global"
	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/xuri/excelize/v2"
	"golang.org/x/crypto/bcrypt"
	"gorm.io/gorm"
)

type UserServiceExtend struct{}

// ListPagedUsers 用户分页列表
func (userService *UserServiceExtend) ListPagedUsers(queryParams query.UserPageQuery) (result vo.PageResult[vo.UserPageVO], err error) {
	// 初始化分页参数
	pageNum := queryParams.PageNum
	pageSize := queryParams.PageSize
	if pageNum <= 0 {
		pageNum = 1
	}
	if pageSize <= 0 {
		pageSize = 10
	}

	// 构建查询
	db := global.DB.Table("sys_user u").
		Select("u.id, u.username, u.nickname, u.mobile, u.gender, u.avatar, u.status, u.email, d.name as dept_name, GROUP_CONCAT(r.name) as role_names, u.create_time").
		Joins("LEFT JOIN sys_dept d ON u.dept_id = d.id").
		Joins("LEFT JOIN sys_user_role sur ON u.id = sur.user_id").
		Joins("LEFT JOIN sys_role r ON sur.role_id = r.id").
		Where("u.deleted = 0 AND u.username != 'root'").
		Group("u.id")

	// 添加查询条件
	if queryParams.Keywords != "" {
		keyword := "%" + queryParams.Keywords + "%"
		db = db.Where("u.username LIKE ? OR u.nickname LIKE ? OR u.mobile LIKE ?", keyword, keyword, keyword)
	}
	if queryParams.Status != nil {
		db = db.Where("u.status = ?", *queryParams.Status)
	}
	if queryParams.DeptId != nil {
		// 获取部门及其所有子部门的ID列表
		deptIds, err := userService.getDeptTreeIds(*queryParams.DeptId)
		if err != nil {
			return result, err
		}
		db = db.Where("u.dept_id IN ?", deptIds)
	}
	if queryParams.StartTime != "" {
		db = db.Where("u.create_time >= ?", queryParams.StartTime)
	}
	if queryParams.EndTime != "" {
		db = db.Where("u.create_time <= ?", queryParams.EndTime)
	}

	// 查询总数
	var total int64
	err = db.Count(&total).Error
	if err != nil {
		return result, err
	}

	// 分页查询
	var userBOs []bo.UserBO
	err = db.Offset((pageNum - 1) * pageSize).Limit(pageSize).Find(&userBOs).Error
	if err != nil {
		return result, err
	}

	// 转换为VO
	var userPageVOs []vo.UserPageVO
	for _, userBO := range userBOs {
		genderLabel := ""
		switch userBO.Gender {
		case 1:
			genderLabel = "男"
		case 2:
			genderLabel = "女"
		default:
			genderLabel = "未知"
		}

		userPageVO := vo.UserPageVO{
			ID:          userBO.ID,
			Username:    userBO.Username,
			Nickname:    userBO.Nickname,
			Mobile:      userBO.Mobile,
			GenderLabel: genderLabel,
			Avatar:      userBO.Avatar,
			Email:       userBO.Email,
			Status:      userBO.Status,
			DeptName:    userBO.DeptName,
			RoleNames:   userBO.RoleNames,
			CreateTime:  userBO.CreateTime,
		}
		userPageVOs = append(userPageVOs, userPageVO)
	}

	// 构造分页结果
	result.List = userPageVOs
	result.Total = total
	result.PageNum = pageNum
	result.PageSize = pageSize

	return result, nil
}

// GetUserFormData 获取用户表单数据
func (userService *UserServiceExtend) GetUserFormData(userId int64) (userFormBO bo.UserFormBO, err error) {
	err = global.DB.Table("sys_user").
		Select("id, username, nickname, mobile, gender, avatar, email, status, dept_id").
		Where("id = ? AND deleted = 0", userId).
		First(&userFormBO).Error

	// 用户不存在时返回空对象（与Java行为保持一致）
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return userFormBO, nil
		}
		return userFormBO, err
	}

	// 查询用户角色ID列表
	var roleIds []int64
	err = global.DB.Table("sys_user_role").
		Where("user_id = ?", userId).
		Pluck("role_id", &roleIds).Error

	if err != nil {
		return userFormBO, err
	}

	userFormBO.RoleIds = roleIds
	return userFormBO, nil
}

// SaveUser 新增用户
func (userService *UserServiceExtend) SaveUser(userFormBO bo.UserFormBO) (err error) {
	// 检查用户名是否已存在
	var count int64
	err = global.DB.Model(&model.SysUser{}).
		Where("username = ? AND deleted = 0", userFormBO.Username).
		Count(&count).Error

	if err != nil {
		return err
	}

	if count > 0 {
		return errors.New("用户名已存在")
	}

	// 加密默认密码
	defaultPassword := "123456"
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(defaultPassword), bcrypt.DefaultCost)
	if err != nil {
		return err
	}

	// 创建用户实体
	sysUser := model.SysUser{
		Username: userFormBO.Username,
		Nickname: userFormBO.Nickname,
		Gender:   userFormBO.Gender,
		DeptID:   userFormBO.DeptID,
		Avatar:   userFormBO.Avatar,
		Mobile:   userFormBO.Mobile,
		Status:   userFormBO.Status,
		Email:    userFormBO.Email,
		// 设置默认加密密码
		Password: string(hashedPassword),
		// Deleted字段默认为0
	}

	// 设置创建时间
	sysUser.CreatedAt = time.Now()
	sysUser.UpdatedAt = time.Now()

	// 开启事务
	tx := global.DB.Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	// 插入用户
	err = tx.Create(&sysUser).Error
	if err != nil {
		tx.Rollback()
		return err
	}

	// 保存用户角色关联
	if len(userFormBO.RoleIds) > 0 {
		var userRoles []map[string]interface{}
		for _, roleId := range userFormBO.RoleIds {
			userRole := map[string]interface{}{
				"user_id": sysUser.ID,
				"role_id": roleId,
			}
			userRoles = append(userRoles, userRole)
		}
		if len(userRoles) > 0 {
			err = tx.Table("sys_user_role").CreateInBatches(userRoles, len(userRoles)).Error
			if err != nil {
				tx.Rollback()
				return err
			}
		}
	}

	// 提交事务
	return tx.Commit().Error
}

// UpdateUser 更新用户
func (userService *UserServiceExtend) UpdateUser(userId int64, userFormBO bo.UserFormBO) (err error) {
	// 注意：用户名创建后不可修改，这里移除用户名唯一性校验和更新

	// 开启事务
	tx := global.DB.Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	// 更新用户信息（排除 username）
	updates := map[string]interface{}{
		"nickname":    userFormBO.Nickname,
		"gender":      userFormBO.Gender,
		"dept_id":     userFormBO.DeptID,
		"avatar":      userFormBO.Avatar,
		"mobile":      userFormBO.Mobile,
		"status":      userFormBO.Status,
		"email":       userFormBO.Email,
		"update_time": time.Now(),
	}

	err = tx.Model(&model.SysUser{}).
		Where("id = ? AND deleted = 0", userId).
		Updates(updates).Error

	if err != nil {
		tx.Rollback()
		return err
	}

	// 更新用户角色关联
	// 先删除原有角色关联
	err = tx.Table("sys_user_role").
		Where("user_id = ?", userId).
		Delete(nil).Error

	if err != nil {
		tx.Rollback()
		return err
	}

	// 再插入新的角色关联
	if len(userFormBO.RoleIds) > 0 {
		var userRoles []map[string]interface{}
		for _, roleId := range userFormBO.RoleIds {
			userRole := map[string]interface{}{
				"user_id": userId,
				"role_id": roleId,
			}
			userRoles = append(userRoles, userRole)
		}
		if len(userRoles) > 0 {
			err = tx.Table("sys_user_role").CreateInBatches(userRoles, len(userRoles)).Error
			if err != nil {
				tx.Rollback()
				return err
			}
		}
	}

	// 提交事务
	return tx.Commit().Error
}

// DeleteUsers 删除用户
func (userService *UserServiceExtend) DeleteUsers(ids string) (err error) {
	if ids == "" {
		return errors.New("删除的用户数据为空")
	}

	// 解析ID列表
	idStrings := strings.Split(ids, ",")
	var idList []int64
	for _, idStr := range idStrings {
		id, err := strconv.ParseInt(idStr, 10, 64)
		if err != nil {
			return errors.New("用户ID格式不正确")
		}
		idList = append(idList, id)
	}

	// 检查是否包含超级管理员
	var count int64
	err = global.DB.Model(&model.SysUser{}).
		Where("id IN ? AND username = ?", idList, "root").
		Count(&count).Error
	if err != nil {
		return err
	}
	if count > 0 {
		return errors.New("超级管理员不能删除")
	}

	// 检查是否删除自己（假设从上下文获取当前用户ID）
	// 这里暂时留空，实际使用时需要从上下文获取

	// 开启事务
	tx := global.DB.Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	// 逻辑删除用户
	err = tx.Model(&model.SysUser{}).
		Where("id IN ?", idList).
		Updates(map[string]interface{}{
			"deleted":     1,
			"update_time": time.Now(),
		}).Error

	if err != nil {
		tx.Rollback()
		return err
	}

	// 提交事务
	return tx.Commit().Error
}

// UpdatePassword 修改用户密码
func (userService *UserServiceExtend) UpdatePassword(userId int64, password string) (err error) {
	// 检查用户是否存在
	var count int64
	err = global.DB.Model(&model.SysUser{}).
		Where("id = ? AND deleted = 0", userId).
		Count(&count).Error
	if err != nil {
		return err
	}
	if count == 0 {
		return errors.New("用户不存在")
	}

	// 加密新密码
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return err
	}

	// 开启事务
	tx := global.DB.Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	err = tx.Model(&model.SysUser{}).
		Where("id = ? AND deleted = 0", userId).
		Updates(map[string]interface{}{
			"password":    string(hashedPassword),
			"update_time": time.Now(),
		}).Error

	if err != nil {
		tx.Rollback()
		return err
	}

	// 提交事务
	return tx.Commit().Error
}

// UpdateUserStatus 修改用户状态
func (userService *UserServiceExtend) UpdateUserStatus(userId int64, status int8) (err error) {
	// 检查是否是超级管理员
	var count int64
	err = global.DB.Model(&model.SysUser{}).
		Where("id = ? AND username = ?", userId, "root").
		Count(&count).Error
	if err != nil {
		return err
	}
	if count > 0 {
		return errors.New("超级管理员不能修改状态")
	}

	// 开启事务
	tx := global.DB.Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	err = tx.Model(&model.SysUser{}).
		Where("id = ? AND deleted = 0", userId).
		Updates(map[string]interface{}{
			"status":      status,
			"update_time": time.Now(),
		}).Error

	if err != nil {
		tx.Rollback()
		return err
	}

	// 提交事务
	return tx.Commit().Error
}

// GetCurrentUserInfo 获取当前登录用户信息
func (userService *UserServiceExtend) GetCurrentUserInfo(username string) (userInfoVO vo.UserInfoVO, err error) {
	// 获取登录用户基础信息
	var user model.SysUser
	err = global.DB.Model(&model.SysUser{}).
		Select("id, username, nickname, avatar").
		Where("username = ? AND deleted = 0", username).
		First(&user).Error

	if err != nil {
		return userInfoVO, err
	}

	userInfoVO.UserId = user.ID
	userInfoVO.Username = user.Username
	userInfoVO.Nickname = user.Nickname
	userInfoVO.Avatar = user.Avatar

	// 查询用户角色编码集合
	var roles []string
	err = global.DB.Table("sys_user u").
		Select("r.code").
		Joins("LEFT JOIN sys_user_role ur ON u.id = ur.user_id").
		Joins("LEFT JOIN sys_role r ON ur.role_id = r.id").
		Where("u.username = ? AND u.deleted = 0 AND r.code IS NOT NULL", username).
		Pluck("r.code", &roles).Error

	if err != nil {
		return userInfoVO, err
	}

	userInfoVO.Roles = roles

	// 查询用户权限标识集合
	if len(roles) > 0 {
		var perms []string
		err = global.DB.Table("sys_menu m").
			Select("DISTINCT m.perm").
			Joins("INNER JOIN sys_role_menu rm ON m.id = rm.menu_id").
			Joins("INNER JOIN sys_role r ON r.id = rm.role_id").
			Where("m.type = ? AND m.perm IS NOT NULL", 4). // 4表示按钮类型
			Where("r.code IN ?", roles).
			Pluck("m.perm", &perms).Error

		if err != nil {
			return userInfoVO, err
		}

		userInfoVO.Perms = perms
	}

	return userInfoVO, nil
}

// ListExportUsers 获取导出用户列表
func (userService *UserServiceExtend) ListExportUsers(queryParams query.UserPageQuery) (userExportVOs []vo.UserExportVO, err error) {
	// 构建查询
	db := global.DB.Table("sys_user u").
		Select("u.username, u.nickname, u.mobile, u.email, u.status, " +
			"CASE u.gender WHEN 1 THEN '男' WHEN 2 THEN '女' ELSE '未知' END as gender, " +
			"d.name as dept_name, u.create_time").
		Joins("LEFT JOIN sys_dept d ON u.dept_id = d.id").
		Where("u.deleted = 0 AND u.username != 'root'")

	// 添加查询条件
	if queryParams.Keywords != "" {
		keyword := "%" + queryParams.Keywords + "%"
		db = db.Where("u.username LIKE ? OR u.nickname LIKE ? OR u.mobile LIKE ?", keyword, keyword, keyword)
	}
	if queryParams.Status != nil {
		db = db.Where("u.status = ?", *queryParams.Status)
	}
	if queryParams.DeptId != nil {
		db = db.Where("u.dept_id = ?", queryParams.DeptId)
	}

	// 查询数据
	var exportData []struct {
		Username   string    `json:"username"`
		Nickname   string    `json:"nickname"`
		DeptName   string    `json:"dept_name"`
		Gender     string    `json:"gender"`
		Mobile     string    `json:"mobile"`
		Email      string    `json:"email"`
		Status     int8      `json:"status"`
		CreateTime time.Time `json:"create_time"`
	}

	err = db.Find(&exportData).Error
	if err != nil {
		return userExportVOs, err
	}

	// 转换为VO
	for _, data := range exportData {
		statusLabel := ""
		if data.Status == 1 {
			statusLabel = "启用"
		} else {
			statusLabel = "禁用"
		}

		userExportVO := vo.UserExportVO{
			Username:    data.Username,
			Nickname:    data.Nickname,
			DeptName:    data.DeptName,
			Gender:      data.Gender,
			Mobile:      data.Mobile,
			Email:       data.Email,
			StatusLabel: statusLabel,
			CreateTime:  data.CreateTime,
		}
		userExportVOs = append(userExportVOs, userExportVO)
	}

	return userExportVOs, nil
}

// getDeptTreeIds 获取部门及其所有子部门的ID列表
func (userService *UserServiceExtend) getDeptTreeIds(deptId int64) ([]int64, error) {
	var allDeptIds []int64
	allDeptIds = append(allDeptIds, deptId)

	// 递归查询所有子部门
	var children []int64
	err := userService.findChildDeptIds(deptId, &children)
	if err != nil {
		return nil, err
	}
	allDeptIds = append(allDeptIds, children...)

	return allDeptIds, nil
}

// findChildDeptIds 递归查找子部门ID
func (userService *UserServiceExtend) findChildDeptIds(parentId int64, result *[]int64) error {
	var children []int64
	err := global.DB.Table("sys_dept").
		Where("parent_id = ? AND deleted = 0", parentId).
		Pluck("id", &children).Error
	if err != nil {
		return err
	}

	for _, childId := range children {
		*result = append(*result, childId)
		// 递归查找子部门的子部门
		err := userService.findChildDeptIds(childId, result)
		if err != nil {
			return err
		}
	}

	return nil
}

// DownloadImportTemplate 下载导入模板
func (userService *UserServiceExtend) DownloadImportTemplate() (string, error) {
	f := excelize.NewFile()
	defer func() {
		if err := f.Close(); err != nil {
			fmt.Printf("关闭Excel文件失败: %v\n", err)
		}
	}()

	// 设置工作表名称
	sheetName := "用户导入"
	if err := f.SetSheetName("Sheet1", sheetName); err != nil {
		return "", err
	}

	// 设置表头
	headers := []string{"用户名*", "昵称*", "性别", "部门名称", "手机号", "邮箱", "状态"}
	for i, header := range headers {
		cell := fmt.Sprintf("%c1", 'A'+i)
		if err := f.SetCellValue(sheetName, cell, header); err != nil {
			return "", err
		}
	}

	// 设置示例数据
	examples := [][]string{
		{"user001", "测试用户1", "男", "研发部", "13800138001", "user001@example.com", "启用"},
		{"user002", "测试用户2", "女", "测试部", "13800138002", "user002@example.com", "启用"},
	}

	for i, example := range examples {
		row := i + 2
		for j, value := range example {
			cell := fmt.Sprintf("%c%d", 'A'+j, row)
			if err := f.SetCellValue(sheetName, cell, value); err != nil {
				return "", err
			}
		}
	}

	// 创建临时目录
	tempDir := filepath.Join(os.TempDir(), "dehaze")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", err
	}

	// 生成临时文件路径
	tempFile := filepath.Join(tempDir, fmt.Sprintf("user_import_template_%d.xlsx", time.Now().Unix()))

	// 保存文件
	if err := f.SaveAs(tempFile); err != nil {
		return "", err
	}

	return tempFile, nil
}

// ImportUsers 导入用户
func (userService *UserServiceExtend) ImportUsers(file io.Reader) (vo.ImportResultVO, error) {
	result := vo.ImportResultVO{}

	// 打开Excel文件
	f, err := excelize.OpenReader(file)
	if err != nil {
		return result, err
	}
	defer func() {
		if err := f.Close(); err != nil {
			fmt.Printf("关闭Excel文件失败: %v\n", err)
		}
	}()

	// 获取所有行数据
	rows, err := f.GetRows("用户导入")
	if err != nil {
		return result, err
	}

	// 没有数据
	if len(rows) <= 1 {
		return result, nil
	}

	// 开启事务
	tx := global.DB.Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	// 逐行处理数据（跳过表头）
	successCount := 0
	failedCount := 0
	var failures []vo.ImportFailureVO

	for i, row := range rows[1:] {
		rowNum := i + 2 // Excel行号从2开始
		result.Total++

		// 验证必填字段
		if len(row) < 2 || row[0] == "" || row[1] == "" {
			failedCount++
			failures = append(failures, vo.ImportFailureVO{
				Row:     rowNum,
				Message: "用户名和昵称为必填项",
			})
			continue
		}

		username := strings.TrimSpace(row[0])
		nickname := strings.TrimSpace(row[1])
		gender := strings.TrimSpace(row[2])
		deptName := strings.TrimSpace(row[3])
		mobile := strings.TrimSpace(row[4])
		email := strings.TrimSpace(row[5])
		statusStr := strings.TrimSpace(row[6])

		// 检查用户名是否已存在
		var count int64
		err = tx.Model(&model.SysUser{}).
			Where("username = ? AND deleted = 0", username).
			Count(&count).Error
		if err != nil {
			failedCount++
			failures = append(failures, vo.ImportFailureVO{
				Row:      rowNum,
				Username: username,
				Message:  "检查用户名失败",
			})
			continue
		}

		if count > 0 {
			failedCount++
			failures = append(failures, vo.ImportFailureVO{
				Row:      rowNum,
				Username: username,
				Message:  "用户名已存在",
			})
			continue
		}

		// 解析性别
		var genderInt int8
		if gender == "男" {
			genderInt = 1
		} else if gender == "女" {
			genderInt = 2
		} else {
			genderInt = 0 // 未知
		}

		// 解析状态
		var status int8 = 1 // 默认启用
		if statusStr == "禁用" || statusStr == "0" {
			status = 0
		}

		// 查询部门ID
		var deptID int64
		if deptName != "" {
			err = tx.Table("sys_dept").
				Where("name = ? AND deleted = 0", deptName).
				Pluck("id", &deptID).Error
			if err != nil {
				// 部门不存在不影响导入，设置0
				deptID = 0
			}
		}

		// 加密默认密码
		defaultPassword := "123456"
		hashedPassword, err := bcrypt.GenerateFromPassword([]byte(defaultPassword), bcrypt.DefaultCost)
		if err != nil {
			failedCount++
			failures = append(failures, vo.ImportFailureVO{
				Row:      rowNum,
				Username: username,
				Message:  "密码加密失败",
			})
			continue
		}

		// 创建用户
		sysUser := model.SysUser{
			Username: username,
			Nickname: nickname,
			Gender:   genderInt,
			DeptID:   deptID,
			Mobile:   mobile,
			Email:    email,
			Status:   status,
			Password: string(hashedPassword),
		}
		sysUser.CreatedAt = time.Now()
		sysUser.UpdatedAt = time.Now()

		err = tx.Create(&sysUser).Error
		if err != nil {
			failedCount++
			failures = append(failures, vo.ImportFailureVO{
				Row:      rowNum,
				Username: username,
				Message:  "保存用户失败",
			})
			continue
		}

		successCount++
	}

	result.Success = successCount
	result.Failed = failedCount
	result.Failures = failures

	// 提交事务
	if err := tx.Commit().Error; err != nil {
		return result, err
	}

	return result, nil
}

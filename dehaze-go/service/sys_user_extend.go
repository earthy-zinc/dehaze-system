package service

import (
	"errors"
	"strconv"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/model/vo"
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
		Select("u.id, u.username, u.nickname, u.mobile, u.gender, u.avatar, u.status, d.name as dept_name, GROUP_CONCAT(r.name) as role_names, u.create_time").
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
		// 注意：这里简化处理，实际应该根据树形结构查询
		db = db.Where("u.dept_id = ?", queryParams.DeptId)
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

	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return userFormBO, errors.New("用户不存在")
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
	// 检查用户名是否已存在（排除当前用户）
	var count int64
	err = global.DB.Model(&model.SysUser{}).
		Where("username = ? AND id != ? AND deleted = 0", userFormBO.Username, userId).
		Count(&count).Error

	if err != nil {
		return err
	}

	if count > 0 {
		return errors.New("用户名已存在")
	}

	// 开启事务
	tx := global.DB.Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	// 更新用户信息
	updates := map[string]interface{}{
		"username":   userFormBO.Username,
		"nickname":   userFormBO.Nickname,
		"gender":     userFormBO.Gender,
		"dept_id":    userFormBO.DeptID,
		"avatar":     userFormBO.Avatar,
		"mobile":     userFormBO.Mobile,
		"status":     userFormBO.Status,
		"email":      userFormBO.Email,
		"updated_at": time.Now(),
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
			"deleted":   1,
			"UpdatedAt": time.Now(),
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
			"password":  string(hashedPassword),
			"UpdatedAt": time.Now(),
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
			"status":    status,
			"UpdatedAt": time.Now(),
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
		Select("u.username, u.nickname, u.mobile, " +
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
		CreateTime time.Time `json:"create_time"`
	}

	err = db.Find(&exportData).Error
	if err != nil {
		return userExportVOs, err
	}

	// 转换为VO
	for _, data := range exportData {
		userExportVO := vo.UserExportVO{
			Username:   data.Username,
			Nickname:   data.Nickname,
			DeptName:   data.DeptName,
			Gender:     data.Gender,
			Mobile:     data.Mobile,
			Email:      data.Email,
			CreateTime: data.CreateTime,
		}
		userExportVOs = append(userExportVOs, userExportVO)
	}

	return userExportVOs, nil
}

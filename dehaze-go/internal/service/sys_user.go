package service

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/global"
	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
	"github.com/xuri/excelize/v2"
	"golang.org/x/crypto/bcrypt"
)

var ErrUserNotFound = errors.New("用户不存在")
var ErrInvalidPassword = errors.New("密码错误")

// UserService 用户服务
type UserService struct {
	userRepo repository.IUserRepository
	roleRepo repository.IRoleRepository
}

// NewUserService 创建用户服务实例
func NewUserService(userRepo repository.IUserRepository, roleRepo repository.IRoleRepository) *UserService {
	return &UserService{
		userRepo: userRepo,
		roleRepo: roleRepo,
	}
}

// getRepo 获取 Repository（兼容零值实例）
func (s *UserService) getRepo() repository.IUserRepository {
	if s.userRepo != nil {
		return s.userRepo
	}
	// 回退：使用 global.DB 创建临时 Repository
	return repository.NewUserRepository(global.DB)
}

// Login 用户登录
func (s *UserService) Login(ctx context.Context, u *model.SysUser) (*model.UserAuthInfo, error) {
	if u == nil {
		return nil, errors.New("用户信息不能为空")
	}

	inputPassword := u.Password
	repo := s.getRepo()

	user, err := repo.FindByUsername(ctx, u.Username)
	if err != nil {
		return nil, err
	}
	if user == nil {
		return nil, ErrUserNotFound
	}

	if err := bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(inputPassword)); err != nil {
		return nil, ErrInvalidPassword
	}

	return s.GetUserAuthInfo(ctx, u.Username)
}

// GetUserAuthInfo 根据用户名获取认证信息
func (s *UserService) GetUserAuthInfo(ctx context.Context, username string) (*model.UserAuthInfo, error) {
	repo := s.getRepo()
	authInfo, err := repo.FindUserAuthInfo(ctx, username)
	if err != nil {
		return nil, err
	}
	if authInfo == nil {
		return nil, ErrUserNotFound
	}
	return authInfo, nil
}

// ====================
// IUserService 接口实现
// ====================

// GetPage 用户分页列表
func (s *UserService) GetPage(ctx context.Context, q *query.UserPageQuery) (*vo.PageResult[vo.UserPageVO], error) {
	repo := s.getRepo()

	// 构建查询
	if q.DeptId != nil {
		deptIds, err := s.getDeptTreeIds(*q.DeptId)
		if err != nil {
			return nil, err
		}
		q.DeptIds = deptIds
	}

	// 使用 Repository 分页查询
	result, err := repo.FindPage(ctx, q)
	if err != nil {
		return nil, err
	}

	return result, nil
}

// GetByID 根据 ID 获取用户
func (s *UserService) GetByID(ctx context.Context, id int64) (*vo.UserPageVO, error) {
	repo := s.getRepo()

	user, err := repo.FindByID(ctx, id)
	if err != nil {
		return nil, err
	}
	if user == nil {
		return nil, ErrUserNotFound
	}

	// 转换为 VO
	var genderLabel string
	switch user.Gender {
	case 1:
		genderLabel = "男"
	case 2:
		genderLabel = "女"
	default:
		genderLabel = "未知"
	}

	userVO := vo.UserPageVO{
		ID:          user.ID,
		Username:    user.Username,
		Nickname:    user.Nickname,
		Mobile:      user.Mobile,
		GenderLabel: genderLabel,
		Avatar:      user.Avatar,
		Email:       user.Email,
		Status:      user.Status,
		CreateTime:  user.CreatedAt,
	}

	return &userVO, nil
}

// GetFormData 获取用户表单数据
func (s *UserService) GetFormData(ctx context.Context, id int64) (*bo.UserFormBO, error) {
	repo := s.getRepo()
	return repo.GetFormData(ctx, id)
}

// Create 创建用户
func (s *UserService) Create(ctx context.Context, form *bo.UserFormBO) error {
	repo := s.getRepo()

	// 检查用户名是否已存在
	exists, err := repo.ExistsByUsername(ctx, form.Username)
	if err != nil {
		return err
	}
	if exists {
		return errors.New("用户名已存在")
	}

	// 加密默认密码
	defaultPassword := "123456"
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(defaultPassword), bcrypt.DefaultCost)
	if err != nil {
		return err
	}

	// 创建用户实体
	user := &model.SysUser{
		Username: form.Username,
		Nickname: form.Nickname,
		Gender:   form.Gender,
		DeptID:   form.DeptID,
		Avatar:   form.Avatar,
		Mobile:   form.Mobile,
		Status:   form.Status,
		Email:    form.Email,
		Password: string(hashedPassword),
	}
	user.CreatedAt = time.Now()
	user.UpdatedAt = time.Now()

	// 开启事务
	tx := global.DB.Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	// 创建用户
	if err := tx.Create(user).Error; err != nil {
		tx.Rollback()
		return err
	}

	// 保存用户角色关联
	if len(form.RoleIds) > 0 {
		var userRoles []map[string]interface{}
		for _, roleId := range form.RoleIds {
			userRoles = append(userRoles, map[string]interface{}{
				"user_id": user.ID,
				"role_id": roleId,
			})
		}
		if len(userRoles) > 0 {
			if err := tx.Table("sys_user_role").CreateInBatches(userRoles, len(userRoles)).Error; err != nil {
				tx.Rollback()
				return err
			}
		}
	}

	return tx.Commit().Error
}

// Update 更新用户
func (s *UserService) Update(ctx context.Context, id int64, form *bo.UserFormBO) error {
	// 开启事务
	tx := global.DB.Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	// 更新用户信息（排除 username）
	updates := map[string]interface{}{
		"nickname":    form.Nickname,
		"gender":      form.Gender,
		"dept_id":     form.DeptID,
		"avatar":      form.Avatar,
		"mobile":      form.Mobile,
		"status":      form.Status,
		"email":       form.Email,
		"update_time": time.Now(),
	}

	if err := tx.Model(&model.SysUser{}).
		Where("id = ? AND deleted = 0", id).
		Updates(updates).Error; err != nil {
		tx.Rollback()
		return err
	}

	// 更新用户角色关联
	// 先删除原有角色关联
	if err := tx.Table("sys_user_role").
		Where("user_id = ?", id).
		Delete(nil).Error; err != nil {
		tx.Rollback()
		return err
	}

	// 再插入新的角色关联
	if len(form.RoleIds) > 0 {
		var userRoles []map[string]interface{}
		for _, roleId := range form.RoleIds {
			userRoles = append(userRoles, map[string]interface{}{
				"user_id": id,
				"role_id": roleId,
			})
		}
		if len(userRoles) > 0 {
			if err := tx.Table("sys_user_role").CreateInBatches(userRoles, len(userRoles)).Error; err != nil {
				tx.Rollback()
				return err
			}
		}
	}

	return tx.Commit().Error
}

// Delete 删除用户（支持批量）
func (s *UserService) Delete(ctx context.Context, ids []int64) error {
	repo := s.getRepo()

	// 检查是否包含超级管理员
	for _, id := range ids {
		user, err := repo.FindByID(ctx, id)
		if err != nil {
			return err
		}
		if user != nil && user.Username == "root" {
			return errors.New("超级管理员不能删除")
		}
	}

	return repo.Delete(ctx, ids)
}

// ResetPassword 重置用户密码
func (s *UserService) ResetPassword(ctx context.Context, id int64) error {
	repo := s.getRepo()

	// 检查用户是否存在
	user, err := repo.FindByID(ctx, id)
	if err != nil {
		return err
	}
	if user == nil {
		return errors.New("用户不存在")
	}

	// 加密默认密码
	defaultPassword := "123456"
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(defaultPassword), bcrypt.DefaultCost)
	if err != nil {
		return err
	}

	return repo.UpdatePassword(ctx, id, string(hashedPassword))
}

// UpdateStatus 更新用户状态
func (s *UserService) UpdateStatus(ctx context.Context, id int64, status int8) error {
	repo := s.getRepo()

	// 检查是否是超级管理员
	user, err := repo.FindByID(ctx, id)
	if err != nil {
		return err
	}
	if user != nil && user.Username == "root" {
		return errors.New("超级管理员不能修改状态")
	}

	return repo.UpdateStatus(ctx, id, status)
}

// GetCurrentUserInfo 获取当前登录用户信息
func (s *UserService) GetCurrentUserInfo(ctx context.Context, userID int64) (*vo.UserInfoVO, error) {
	repo := s.getRepo()

	// 获取用户基础信息
	user, err := repo.FindByID(ctx, userID)
	if err != nil {
		return nil, err
	}
	if user == nil {
		return nil, ErrUserNotFound
	}

	userInfoVO := vo.UserInfoVO{
		UserId:   user.ID,
		Username: user.Username,
		Nickname: user.Nickname,
		Avatar:   user.Avatar,
	}

	// 查询用户角色编码集合
	roleIds, err := repo.GetUserRoleIDs(ctx, userID)
	if err != nil {
		return &userInfoVO, nil // 角色查询失败不影响返回用户信息
	}

	// 获取角色编码
	var roles []string
	for _, roleId := range roleIds {
		role, err := s.roleRepo.FindByID(ctx, roleId)
		if err == nil && role != nil && role.Code != "" {
			roles = append(roles, role.Code)
		}
	}
	userInfoVO.Roles = roles

	// 查询用户权限标识集合
	if len(roles) > 0 {
		// TODO: 实现权限查询逻辑
		// 暂时返回空权限列表
		userInfoVO.Perms = []string{}
	}

	return &userInfoVO, nil
}

// ImportUsers 导入用户
func (s *UserService) ImportUsers(ctx context.Context, data []vo.UserImportVO) (*vo.ImportResultVO, error) {
	result := vo.ImportResultVO{Total: len(data)}

	// 开启事务
	tx := global.DB.Begin()
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	successCount := 0
	failedCount := 0
	var failures []vo.ImportFailureVO

	for i, item := range data {
		rowNum := i + 1

		// 验证必填字段
		if item.Username == "" || item.Nickname == "" {
			failedCount++
			failures = append(failures, vo.ImportFailureVO{
				Row:      rowNum,
				Username: item.Username,
				Message:  "用户名和昵称为必填项",
			})
			continue
		}

		// 检查用户名是否已存在
		var count int64
		err := tx.Model(&model.SysUser{}).
			Where("username = ? AND deleted = 0", item.Username).
			Count(&count).Error
		if err != nil {
			failedCount++
			failures = append(failures, vo.ImportFailureVO{
				Row:      rowNum,
				Username: item.Username,
				Message:  "检查用户名失败",
			})
			continue
		}

		if count > 0 {
			failedCount++
			failures = append(failures, vo.ImportFailureVO{
				Row:      rowNum,
				Username: item.Username,
				Message:  "用户名已存在",
			})
			continue
		}

		// 解析性别
		var genderInt int8
		if item.Gender == "男" {
			genderInt = 1
		} else if item.Gender == "女" {
			genderInt = 2
		} else {
			genderInt = 0 // 未知
		}

		// 解析状态
		var status int8 = 1 // 默认启用
		if item.Status == "禁用" || item.Status == "0" {
			status = 0
		}

		// 查询部门ID
		var deptID int64
		if item.DeptName != "" {
			err = tx.Table("sys_dept").
				Where("name = ? AND deleted = 0", item.DeptName).
				Pluck("id", &deptID).Error
			if err != nil {
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
				Username: item.Username,
				Message:  "密码加密失败",
			})
			continue
		}

		// 创建用户
		sysUser := model.SysUser{
			Username: item.Username,
			Nickname: item.Nickname,
			Gender:   genderInt,
			DeptID:   deptID,
			Mobile:   item.Mobile,
			Email:    item.Email,
			Status:   status,
			Password: string(hashedPassword),
		}
		sysUser.CreatedAt = time.Now()
		sysUser.UpdatedAt = time.Now()

		if err := tx.Create(&sysUser).Error; err != nil {
			failedCount++
			failures = append(failures, vo.ImportFailureVO{
				Row:      rowNum,
				Username: item.Username,
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
		return &result, err
	}

	return &result, nil
}

// ExportUsers 导出用户
func (s *UserService) ExportUsers(ctx context.Context, q *query.UserPageQuery) ([]vo.UserExportVO, error) {
	// 构建查询
	db := global.DB.Table("sys_user u").
		Select("u.username, u.nickname, u.mobile, u.email, u.status, " +
			"CASE u.gender WHEN 1 THEN '男' WHEN 2 THEN '女' ELSE '未知' END as gender, " +
			"d.name as dept_name, u.create_time").
		Joins("LEFT JOIN sys_dept d ON u.dept_id = d.id").
		Where("u.deleted = 0 AND u.username != 'root'")

	// 添加查询条件
	if q.Keywords != "" {
		keyword := "%" + q.Keywords + "%"
		db = db.Where("u.username LIKE ? OR u.nickname LIKE ? OR u.mobile LIKE ?", keyword, keyword, keyword)
	}
	if q.Status != nil {
		db = db.Where("u.status = ?", *q.Status)
	}
	if q.DeptId != nil {
		db = db.Where("u.dept_id = ?", q.DeptId)
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

	if err := db.Find(&exportData).Error; err != nil {
		return nil, err
	}

	// 转换为VO
	var userExportVOs []vo.UserExportVO
	for _, data := range exportData {
		statusLabel := ""
		if data.Status == 1 {
			statusLabel = "启用"
		} else {
			statusLabel = "禁用"
		}

		userExportVOs = append(userExportVOs, vo.UserExportVO{
			Username:    data.Username,
			Nickname:    data.Nickname,
			DeptName:    data.DeptName,
			Gender:      data.Gender,
			Mobile:      data.Mobile,
			Email:       data.Email,
			StatusLabel: statusLabel,
			CreateTime:  data.CreateTime,
		})
	}

	return userExportVOs, nil
}

// ====================
// 辅助方法
// ====================

// getDeptTreeIds 获取部门及其所有子部门的ID列表
func (s *UserService) getDeptTreeIds(deptId int64) ([]int64, error) {
	var allDeptIds []int64
	allDeptIds = append(allDeptIds, deptId)

	// 递归查询所有子部门
	var children []int64
	err := s.findChildDeptIds(deptId, &children)
	if err != nil {
		return nil, err
	}
	allDeptIds = append(allDeptIds, children...)

	return allDeptIds, nil
}

// findChildDeptIds 递归查找子部门ID
func (s *UserService) findChildDeptIds(parentId int64, result *[]int64) error {
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
		if err := s.findChildDeptIds(childId, result); err != nil {
			return err
		}
	}

	return nil
}

// DownloadImportTemplate 下载导入模板
func (s *UserService) DownloadImportTemplate() (string, error) {
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

// ImportUsersFromFile 从文件导入用户
func (s *UserService) ImportUsersFromFile(file io.Reader) (*vo.ImportResultVO, error) {
	result := &vo.ImportResultVO{}

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

	// 转换为 UserImportVO 数组
	var data []vo.UserImportVO
	for _, row := range rows[1:] {
		if len(row) < 2 {
			continue
		}

		item := vo.UserImportVO{
			Username: strings.TrimSpace(row[0]),
			Nickname: strings.TrimSpace(row[1]),
			Gender:   strings.TrimSpace(row[2]),
			DeptName: strings.TrimSpace(row[3]),
			Mobile:   strings.TrimSpace(row[4]),
			Email:    strings.TrimSpace(row[5]),
			Status:   strings.TrimSpace(row[6]),
		}
		data = append(data, item)
	}

	return s.ImportUsers(context.Background(), data)
}

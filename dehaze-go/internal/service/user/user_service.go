package user

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	deptrepo "github.com/earthyzinc/dehaze-go/internal/repository/dept"
	menurepo "github.com/earthyzinc/dehaze-go/internal/repository/menu"
	rolerepo "github.com/earthyzinc/dehaze-go/internal/repository/role"
	userrepo "github.com/earthyzinc/dehaze-go/internal/repository/user"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/xuri/excelize/v2"
	"go.uber.org/zap"
	"golang.org/x/crypto/bcrypt"
)

var ErrUserNotFound = common.NewBizError(common.RESOURCE_NOT_FOUND, "用户不存在")
var ErrInvalidPassword = common.NewBizError(common.USERNAME_OR_PASSWORD_ERROR, "密码错误")

// UserService 用户服务
type UserService struct {
	userRepo userrepo.IUserRepository
	roleRepo rolerepo.IRoleRepository
	deptRepo deptrepo.IDeptRepository
	menuRepo menurepo.IMenuRepository
}

// NewUserService 创建用户服务实例
func NewUserService(userRepo userrepo.IUserRepository, roleRepo rolerepo.IRoleRepository, deptRepo deptrepo.IDeptRepository, menuRepo menurepo.IMenuRepository) *UserService {
	return &UserService{
		userRepo: userRepo,
		roleRepo: roleRepo,
		deptRepo: deptRepo,
		menuRepo: menuRepo,
	}
}

// Login 用户登录
func (s *UserService) Login(ctx context.Context, u *model.SysUser) (*model.UserAuthInfo, error) {
	if u == nil {
		return nil, common.NewBizError(common.PARAM_ERROR, "用户信息不能为空")
	}

	inputPassword := u.Password

	// 直接获取认证信息（含用户基本信息、角色、权限），消除冗余的用户表查询
	authInfo, err := s.GetUserAuthInfo(ctx, u.Username)
	if err != nil {
		return nil, err
	}

	if err := bcrypt.CompareHashAndPassword([]byte(authInfo.Password), []byte(inputPassword)); err != nil {
		return nil, ErrInvalidPassword
	}

	return authInfo, nil
}

// GetUserAuthInfo 根据用户名获取认证信息
func (s *UserService) GetUserAuthInfo(ctx context.Context, username string) (*model.UserAuthInfo, error) {
	authInfo, err := s.userRepo.FindUserAuthInfo(ctx, username)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询用户认证信息失败", err)
	}
	if authInfo == nil {
		return nil, ErrUserNotFound
	}
	return authInfo, nil
}

func (s *UserService) GetUserAuthInfoByID(ctx context.Context, userID int64) (*model.UserAuthInfo, error) {
	authInfo, err := s.userRepo.FindUserAuthInfoByID(ctx, userID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询用户认证信息失败", err)
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
	// 构建查询
	if q.DeptId != nil {
		deptIds, err := s.deptRepo.GetSubDeptIDs(ctx, *q.DeptId)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询子部门失败", err)
		}
		q.DeptIds = deptIds
	}

	// 使用 Repository 分页查询
	readResult, err := s.userRepo.FindPage(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "分页查询用户失败", err)
	}
	if readResult == nil {
		return &vo.PageResult[vo.UserPageVO]{List: []vo.UserPageVO{}, Total: 0}, nil
	}

	voList := make([]vo.UserPageVO, 0, len(readResult.List))
	for _, item := range readResult.List {
		voList = append(voList, vo.UserPageVO{
			ID:          item.ID,
			Username:    item.Username,
			Nickname:    item.Nickname,
			Mobile:      item.Mobile,
			GenderLabel: item.GenderLabel,
			Avatar:      item.Avatar,
			Email:       item.Email,
			Status:      item.Status,
			DeptName:    item.DeptName,
			RoleNames:   item.RoleNames,
			CreateTime:  item.CreateTime.Format("2006-01-02"),
		})
	}

	return &vo.PageResult[vo.UserPageVO]{
		List:  voList,
		Total: readResult.Total,
	}, nil
}

// GetByID 根据 ID 获取用户
func (s *UserService) GetByID(ctx context.Context, id int64) (*vo.UserPageVO, error) {
	user, err := s.userRepo.FindByID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询用户失败", err)
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
		CreateTime:  user.CreatedAt.Format("2006-01-02"),
	}

	return &userVO, nil
}

// GetFormData 获取用户表单数据
func (s *UserService) GetFormData(ctx context.Context, id int64) (*bo.UserFormBO, error) {
	form, err := s.userRepo.GetFormData(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询用户表单数据失败", err)
	}
	return form, nil
}

// Create 创建用户
func (s *UserService) Create(ctx context.Context, form *bo.UserFormBO) error {
	// 检查用户名是否已存在
	exists, err := s.userRepo.ExistsByUsername(ctx, form.Username)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "检查用户名是否存在失败", err)
	}
	if exists {
		return common.NewBizError(common.DATA_EXISTS, "用户名已存在")
	}

	// 检查手机号是否已存在
	if form.Mobile != "" {
		exists, err = s.userRepo.ExistsByMobile(ctx, form.Mobile)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "检查手机号是否存在失败", err)
		}
		if exists {
			return common.NewBizError(common.DATA_EXISTS, "手机号已存在")
		}
	}

	// 检查邮箱是否已存在
	if form.Email != "" {
		exists, err = s.userRepo.ExistsByEmail(ctx, form.Email)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "检查邮箱是否存在失败", err)
		}
		if exists {
			return common.NewBizError(common.DATA_EXISTS, "邮箱已存在")
		}
	}

	// 加密默认密码
	defaultPassword := "123456"
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(defaultPassword), bcrypt.DefaultCost)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "密码加密失败", err)
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

	if err := s.userRepo.CreateWithRoles(ctx, user, form.RoleIds); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "创建用户失败", err)
	}
	return nil
}

// Update 更新用户
func (s *UserService) Update(ctx context.Context, id int64, form *bo.UserFormBO) error {
	// 校验用户是否存在
	existingUser, err := s.userRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询用户失败", err)
	}
	if existingUser == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "用户不存在")
	}

	// 校验用户名冲突（如果要修改用户名）
	if form.Username != existingUser.Username {
		exists, err := s.userRepo.ExistsByUsername(ctx, form.Username, id)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "检查用户名是否存在失败", err)
		}
		if exists {
			return common.NewBizError(common.DATA_EXISTS, "用户名已存在")
		}
	}

	// 检查手机号是否已存在（排除当前用户）
	if form.Mobile != "" {
		exists, err := s.userRepo.ExistsByMobile(ctx, form.Mobile, id)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "检查手机号是否存在失败", err)
		}
		if exists {
			return common.NewBizError(common.DATA_EXISTS, "手机号已存在")
		}
	}

	// 检查邮箱是否已存在（排除当前用户）
	if form.Email != "" {
		exists, err := s.userRepo.ExistsByEmail(ctx, form.Email, id)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "检查邮箱是否存在失败", err)
		}
		if exists {
			return common.NewBizError(common.DATA_EXISTS, "邮箱已存在")
		}
	}

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

	if err := s.userRepo.UpdateWithRoles(ctx, id, updates, form.RoleIds); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新用户失败", err)
	}
	return nil
}

// Delete 删除用户（支持批量）
func (s *UserService) Delete(ctx context.Context, ids []int64) error {
	// 检查是否包含超级管理员
	isRoot, err := s.userRepo.ExistsRootInIDs(ctx, ids)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "检查超级管理员失败", err)
	}
	if isRoot {
		return common.NewBizError(common.OPERATION_NOT_ALLOW, "超级管理员不能删除")
	}

	if err := s.userRepo.SoftDeleteWithTime(ctx, ids, time.Now()); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除用户失败", err)
	}
	return nil
}

// UpdatePassword 修改用户密码
func (s *UserService) UpdatePassword(ctx context.Context, id int64, password string) error {
	user, err := s.userRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询用户失败", err)
	}
	if user == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "用户不存在")
	}

	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "密码加密失败", err)
	}

	if err := s.userRepo.UpdatePasswordWithTime(ctx, id, string(hashedPassword), time.Now()); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新密码失败", err)
	}
	return nil
}

// ResetPassword 重置用户密码
func (s *UserService) ResetPassword(ctx context.Context, id int64) error {
	// 检查用户是否存在
	user, err := s.userRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询用户失败", err)
	}
	if user == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "用户不存在")
	}

	// 加密默认密码
	defaultPassword := "123456"
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(defaultPassword), bcrypt.DefaultCost)
	if err != nil {
		return common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "密码加密失败", err)
	}

	if err := s.userRepo.UpdatePasswordWithTime(ctx, id, string(hashedPassword), time.Now()); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新密码失败", err)
	}
	return nil
}

// UpdateStatus 更新用户状态
func (s *UserService) UpdateStatus(ctx context.Context, id int64, status int8) error {
	// 检查是否是超级管理员
	isRoot, err := s.userRepo.ExistsRootInIDs(ctx, []int64{id})
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "检查超级管理员失败", err)
	}
	if isRoot {
		return common.NewBizError(common.OPERATION_NOT_ALLOW, "超级管理员不能修改状态")
	}

	if err := s.userRepo.UpdateStatusWithTime(ctx, id, status, time.Now()); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新用户状态失败", err)
	}
	return nil
}

// GetCurrentUserInfo 获取当前登录用户信息
func (s *UserService) GetCurrentUserInfo(ctx context.Context, userID int64) (*vo.UserInfoVO, error) {
	// 合并查询用户基础信息和角色编码（单次JOIN，消除冗余的用户表查询）
	user, roleCodes, err := s.userRepo.FindUserWithRoleCodesByID(ctx, userID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询用户失败", err)
	}
	if user == nil {
		return nil, ErrUserNotFound
	}

	userInfoVO := vo.UserInfoVO{
		UserId:   user.ID,
		Username: user.Username,
		Nickname: user.Nickname,
		Avatar:   user.Avatar,
		Roles:    roleCodes,
	}

	// 查询用户权限标识集合
	if len(roleCodes) > 0 {
		perms, err := s.menuRepo.FindPermsByRolesWithType(ctx, roleCodes, 4)
		if err != nil {
			return &userInfoVO, nil // 权限查询失败不影响返回用户信息
		}
		userInfoVO.Perms = perms
	}

	return &userInfoVO, nil
}

// ImportUsers 导入用户
func (s *UserService) ImportUsers(ctx context.Context, data []vo.UserImportVO) (*vo.ImportResultVO, error) {
	result := vo.ImportResultVO{Total: len(data)}

	successCount := 0
	failedCount := 0
	var failures []vo.ImportFailureVO

	err := s.userRepo.Transaction(ctx, func(txRepo userrepo.IUserRepository) error {
		// 批量预取部门 ID（避免循环内逐条 FindIDByName 触发 N+1 查询）
		deptNameSet := make(map[string]struct{})
		for _, item := range data {
			if item.DeptName != "" {
				deptNameSet[item.DeptName] = struct{}{}
			}
		}
		deptNameList := make([]string, 0, len(deptNameSet))
		for name := range deptNameSet {
			deptNameList = append(deptNameList, name)
		}
		deptIDMap, err := s.deptRepo.FindIDsByNames(ctx, deptNameList)
		if err != nil {
			return err
		}

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
			exists, err := txRepo.ExistsByUsername(ctx, item.Username)
			if err != nil {
				failedCount++
				failures = append(failures, vo.ImportFailureVO{
					Row:      rowNum,
					Username: item.Username,
					Message:  "检查用户名失败",
				})
				continue
			}

			if exists {
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

			// 从批量预取的映射中获取部门 ID
			var deptID int64
			if item.DeptName != "" {
				deptID = deptIDMap[item.DeptName]
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

			if err := txRepo.Create(ctx, &sysUser); err != nil {
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

		return nil
	})
	if err != nil {
		return &result, common.WrapBizError(common.DATABASE_ERROR, "导入用户事务失败", err)
	}

	result.Success = successCount
	result.Failed = failedCount
	result.Failures = failures

	return &result, nil
}

// ExportUsers 导出用户到Excel文件，返回临时文件路径
func (s *UserService) ExportUsers(ctx context.Context, q *query.UserPageQuery) (string, error) {
	readExports, err := s.userRepo.FindExportUsers(ctx, q)
	if err != nil {
		return "", common.WrapBizError(common.DATABASE_ERROR, "查询导出用户列表失败", err)
	}

	// 创建Excel文件
	f := excelize.NewFile()
	defer func() {
		if err := f.Close(); err != nil {
			logger.Warn("关闭Excel文件失败", zap.Error(err))
		}
	}()

	sheetName := "用户列表"
	if err := f.SetSheetName("Sheet1", sheetName); err != nil {
		return "", common.WrapBizError(common.FILE_UPLOAD_FAILED, "设置工作表名称失败", err)
	}

	// 设置表头
	headers := []string{"用户名", "昵称", "部门", "性别", "手机号", "邮箱", "状态", "创建时间"}
	for i, header := range headers {
		cell := fmt.Sprintf("%c1", 'A'+i)
		if err := f.SetCellValue(sheetName, cell, header); err != nil {
			return "", common.WrapBizError(common.FILE_UPLOAD_FAILED, "设置表头失败", err)
		}
	}

	// 填充数据
	for i, item := range readExports {
		row := i + 2
		createTimeStr := ""
		if !item.CreateTime.IsZero() {
			createTimeStr = item.CreateTime.Format("2006-01-02 15:04:05")
		}
		values := []string{item.Username, item.Nickname, item.DeptName, item.Gender, item.Mobile, item.Email, item.StatusLabel, createTimeStr}
		for j, value := range values {
			cell := fmt.Sprintf("%c%d", 'A'+j, row)
			if err := f.SetCellValue(sheetName, cell, value); err != nil {
				return "", common.WrapBizError(common.FILE_UPLOAD_FAILED, "设置导出数据失败", err)
			}
		}
	}

	// 创建临时目录
	tempDir := filepath.Join(os.TempDir(), "dehaze")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", common.WrapBizError(common.SYSTEM_RESOURCE_ERROR, "创建临时目录失败", err)
	}

	tempFile := filepath.Join(tempDir, fmt.Sprintf("users_export_%d.xlsx", time.Now().Unix()))

	if err := f.SaveAs(tempFile); err != nil {
		return "", common.WrapBizError(common.FILE_UPLOAD_FAILED, "保存导出文件失败", err)
	}

	return tempFile, nil
}

// ====================
// 辅助方法
// ====================

// DownloadImportTemplate 下载导入模板
func (s *UserService) DownloadImportTemplate(ctx context.Context) (string, error) {
	f := excelize.NewFile()
	defer func() {
		if err := f.Close(); err != nil {
			logger.Warn("关闭Excel文件失败", zap.Error(err))
		}
	}()

	// 设置工作表名称
	sheetName := "用户导入"
	if err := f.SetSheetName("Sheet1", sheetName); err != nil {
		return "", common.WrapBizError(common.FILE_UPLOAD_FAILED, "设置工作表名称失败", err)
	}

	// 设置表头
	headers := []string{"用户名*", "昵称*", "性别", "部门名称", "手机号", "邮箱", "状态"}
	for i, header := range headers {
		cell := fmt.Sprintf("%c1", 'A'+i)
		if err := f.SetCellValue(sheetName, cell, header); err != nil {
			return "", common.WrapBizError(common.FILE_UPLOAD_FAILED, "设置表头失败", err)
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
				return "", common.WrapBizError(common.FILE_UPLOAD_FAILED, "设置示例数据失败", err)
			}
		}
	}

	// 创建临时目录
	tempDir := filepath.Join(os.TempDir(), "dehaze")
	if err := os.MkdirAll(tempDir, 0755); err != nil {
		return "", common.WrapBizError(common.SYSTEM_RESOURCE_ERROR, "创建临时目录失败", err)
	}

	// 生成临时文件路径
	tempFile := filepath.Join(tempDir, fmt.Sprintf("user_import_template_%d.xlsx", time.Now().Unix()))

	// 保存文件
	if err := f.SaveAs(tempFile); err != nil {
		return "", common.WrapBizError(common.FILE_UPLOAD_FAILED, "保存模板文件失败", err)
	}

	return tempFile, nil
}

// ImportUsersFromFile 从文件导入用户
func (s *UserService) ImportUsersFromFile(ctx context.Context, file io.Reader) (*vo.ImportResultVO, error) {
	result := &vo.ImportResultVO{}

	// 打开Excel文件
	f, err := excelize.OpenReader(file)
	if err != nil {
		return result, common.WrapBizError(common.FILE_UPLOAD_FAILED, "打开Excel文件失败", err)
	}
	defer func() {
		if err := f.Close(); err != nil {
			logger.Warn("关闭Excel文件失败", zap.Error(err))
		}
	}()

	// 获取所有行数据
	rows, err := f.GetRows("用户导入")
	if err != nil {
		return result, common.WrapBizError(common.FILE_UPLOAD_FAILED, "读取Excel工作表失败", err)
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
		}
		if len(row) > 2 {
			item.Gender = strings.TrimSpace(row[2])
		}
		if len(row) > 3 {
			item.DeptName = strings.TrimSpace(row[3])
		}
		if len(row) > 4 {
			item.Mobile = strings.TrimSpace(row[4])
		}
		if len(row) > 5 {
			item.Email = strings.TrimSpace(row[5])
		}
		if len(row) > 6 {
			item.Status = strings.TrimSpace(row[6])
		}
		data = append(data, item)
	}

	return s.ImportUsers(ctx, data)
}

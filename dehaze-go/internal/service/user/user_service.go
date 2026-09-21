package user

import (
	"context"
	"errors"
	"fmt"
	"time"

	mysql "github.com/go-sql-driver/mysql"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	deptrepo "github.com/earthyzinc/dehaze-go/internal/repository/dept"
	memberrepo "github.com/earthyzinc/dehaze-go/internal/repository/member"
	menurepo "github.com/earthyzinc/dehaze-go/internal/repository/menu"
	rolerepo "github.com/earthyzinc/dehaze-go/internal/repository/role"
	userrepo "github.com/earthyzinc/dehaze-go/internal/repository/user"
	auditlogservice "github.com/earthyzinc/dehaze-go/internal/service/audit_log"
	"github.com/earthyzinc/dehaze-go/internal/service/session"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config"
	"github.com/earthyzinc/dehaze-go/pkg/database"
	"golang.org/x/crypto/bcrypt"
)

// isDuplicateKeyError 判断是否为 MySQL 唯一键冲突（1062）。
// gorm 未开启 TranslateError，故直接识别驱动的 MySQLError。
func isDuplicateKeyError(err error) bool {
	var mysqlErr *mysql.MySQLError
	return errors.As(err, &mysqlErr) && mysqlErr.Number == 1062
}

var ErrUserNotFound = common.NewBizError(common.RESOURCE_NOT_FOUND, "用户不存在")
var ErrInvalidPassword = common.NewBizError(common.USERNAME_OR_PASSWORD_ERROR, "用户名或密码错误")

// UserService 用户服务
type UserService struct {
	userRepo    userrepo.IUserRepository
	roleRepo    rolerepo.IRoleRepository
	deptRepo    deptrepo.IDeptRepository
	menuRepo    menurepo.IMenuRepository
	memberRepo  memberrepo.IMemberRepository
	auditLogSvc *auditlogservice.AuditLogService
}

// NewUserService 创建用户服务实例
func NewUserService(userRepo userrepo.IUserRepository, roleRepo rolerepo.IRoleRepository, deptRepo deptrepo.IDeptRepository, menuRepo menurepo.IMenuRepository, memberRepo memberrepo.IMemberRepository, auditLogSvc *auditlogservice.AuditLogService) *UserService {
	return &UserService{
		userRepo:    userRepo,
		roleRepo:    roleRepo,
		deptRepo:    deptRepo,
		menuRepo:    menuRepo,
		memberRepo:  memberRepo,
		auditLogSvc: auditLogSvc,
	}
}

// Login 用户登录
func (s *UserService) Login(ctx context.Context, u *model.SysUser) (*model.UserAuthInfo, error) {
	if u == nil {
		return nil, common.NewBizError(common.PARAM_ERROR, "用户信息不能为空")
	}

	inputPassword := u.Password

	// 直接获取认证信息（含用户基本信息、角色、权限），消除冗余的用户表查询。
	// 用户不存在与密码错误统一返回 A0210（与 Python 端登录口径一致，防用户名枚举）
	authInfo, err := s.GetUserAuthInfo(ctx, u.Username)
	if err != nil {
		return nil, ErrInvalidPassword
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
			UserType:    item.UserType,
			DeptName:    item.DeptName,
			RoleNames:   item.RoleNames,
			CreateTime:  item.CreateTime.Format("2006-01-02"),
		})
	}

	// 批量聚合会员信息（单次 IN 查询，避免 N+1）
	if len(voList) > 0 {
		userIDs := make([]int64, 0, len(voList))
		for _, v := range voList {
			userIDs = append(userIDs, v.ID)
		}
		members, err := s.memberRepo.FindByUserIDs(ctx, userIDs)
		if err != nil {
			return nil, common.WrapBizError(common.DATABASE_ERROR, "查询用户会员信息失败", err)
		}
		memberMap := make(map[int64]model.SysMember, len(members))
		for _, m := range members {
			memberMap[m.UserID] = m
		}
		for i := range voList {
			voList[i].QuotaUsage = "0/0"
			m, ok := memberMap[voList[i].ID]
			if !ok {
				continue
			}
			voList[i].MemberLevel = &m.LevelCode
			if m.ExpireTime != nil {
				expireTime := m.ExpireTime.Format("2006-01-02 15:04:05")
				voList[i].MemberExpireTime = &expireTime
			}
			quota := m.MonthlyDehazeQuota + m.MonthlyDerainQuota + m.MonthlyDesnowQuota + m.MonthlyLowlightQuota +
				m.MonthlySuperResolutionQuota + m.MonthlyDenoiseQuota + m.MonthlyInpaintQuota + m.MonthlyEvaluateQuota
			used := m.MonthlyDehazeUsed + m.MonthlyDerainUsed + m.MonthlyDesnowUsed + m.MonthlyLowlightUsed +
				m.MonthlySuperResolutionUsed + m.MonthlyDenoiseUsed + m.MonthlyInpaintUsed + m.MonthlyEvaluateUsed
			voList[i].QuotaUsage = fmt.Sprintf("%d/%d", used, quota)
		}
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
		UserType:    user.UserType,
		CreateTime:  user.CreatedAt.Format("2006-01-02"),
	}

	member, err := s.memberRepo.FindByUserID(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询用户会员信息失败", err)
	}
	userVO.QuotaUsage = "0/0"
	if member != nil {
		userVO.MemberLevel = &member.LevelCode
		if member.ExpireTime != nil {
			expireTime := member.ExpireTime.Format("2006-01-02 15:04:05")
			userVO.MemberExpireTime = &expireTime
		}
		quota := member.MonthlyDehazeQuota + member.MonthlyDerainQuota + member.MonthlyDesnowQuota + member.MonthlyLowlightQuota +
			member.MonthlySuperResolutionQuota + member.MonthlyDenoiseQuota + member.MonthlyInpaintQuota + member.MonthlyEvaluateQuota
		used := member.MonthlyDehazeUsed + member.MonthlyDerainUsed + member.MonthlyDesnowUsed + member.MonthlyLowlightUsed +
			member.MonthlySuperResolutionUsed + member.MonthlyDenoiseUsed + member.MonthlyInpaintUsed + member.MonthlyEvaluateUsed
		userVO.QuotaUsage = fmt.Sprintf("%d/%d", used, quota)
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
	defaultPassword := config.GetConfig().System.DefaultPassword
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
		UserType: form.UserType,
		Password: string(hashedPassword),
	}
	if user.UserType == "" {
		user.UserType = "personal"
	}
	// 时间截断到秒（列为 DATETIME 秒精度，同 Register）
	user.CreatedAt = time.Now().Truncate(time.Second)
	user.UpdatedAt = time.Now().Truncate(time.Second)

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

	// 用户名字段只读，不可修改（与 Python 端 update_user_with_roles 口径一致）
	if form.Username != "" && form.Username != existingUser.Username {
		return common.NewBizError(common.OPERATION_NOT_ALLOW, "用户名不可修改")
	}

	// 检查手机号是否已存在（排除当前用户）
	if form.Mobile != "" {
		exists, err := s.userRepo.ExistsByMobile(ctx, form.Mobile, id)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "检查手机号是否存在失败", err)
		}
		if exists {
			return common.NewBizError(common.DATA_EXISTS, "该手机号不可用")
		}
	}

	// 检查邮箱是否已存在（排除当前用户）
	if form.Email != "" {
		exists, err := s.userRepo.ExistsByEmail(ctx, form.Email, id)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "检查邮箱是否存在失败", err)
		}
		if exists {
			return common.NewBizError(common.DATA_EXISTS, "该邮箱不可用")
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
	// userType 可选指定，仅显式传入时更新
	if form.UserType != "" {
		updates["user_type"] = form.UserType
	}

	if err := s.userRepo.UpdateWithRoles(ctx, id, updates, form.RoleIds); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新用户失败", err)
	}
	return nil
}

// Delete 删除用户（支持批量）
func (s *UserService) Delete(ctx context.Context, ids []int64) error {
	// 不可删除自己（当前登录用户视角）
	if operatorID := database.GetUserID(ctx); operatorID != 0 {
		for _, id := range ids {
			if id == operatorID {
				return common.NewBizError(common.OPERATION_NOT_ALLOW, "不可删除自己")
			}
		}
	}

	// 超级管理员受保护，不可删除
	isRoot, err := s.userRepo.ExistsRootInIDs(ctx, ids)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "检查超级管理员失败", err)
	}
	if isRoot {
		return common.NewBizError(common.ROOT_USER_PROTECTED, "超级管理员不可删除")
	}

	if err := s.userRepo.SoftDeleteWithTime(ctx, ids, time.Now()); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除用户失败", err)
	}
	if s.auditLogSvc != nil {
		s.auditLogSvc.RecordAuditAsync(ctx, database.GetUserID(ctx), "user", ids, "delete", "user", nil, nil, database.GetIP(ctx), database.GetUserAgent(ctx))
	}

	// 删除后踢出目标用户全部在线会话（文档 §3.4.3）
	if _, err := session.KickByUserIDs(ctx, ids); err != nil {
		return err
	}
	return nil
}

// validatePasswordComplexity 密码复杂度校验（8-20 位，必须包含字母和数字），与 Python 端 validate_password_complexity 对齐
func validatePasswordComplexity(password string) bool {
	if len(password) < 8 || len(password) > 20 {
		return false
	}
	hasLetter, hasDigit := false, false
	for i := 0; i < len(password); i++ {
		c := password[i]
		switch {
		case (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'):
			hasLetter = true
		case c >= '0' && c <= '9':
			hasDigit = true
		}
	}
	return hasLetter && hasDigit
}

// UpdatePassword 修改用户密码
func (s *UserService) UpdatePassword(ctx context.Context, id int64, password string) error {
	// 密码复杂度校验（A0400），与 Python 端一致：先校验复杂度再查用户存在性
	if !validatePasswordComplexity(password) {
		return common.NewBizError(common.PARAM_ERROR, "密码必须包含字母和数字，8-20位")
	}

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
	if s.auditLogSvc != nil {
		s.auditLogSvc.RecordAuditAsync(ctx, database.GetUserID(ctx), "user", id, "password_change", "user", nil, nil, database.GetIP(ctx), database.GetUserAgent(ctx))
	}

	// 重置后踢出目标用户全部在线会话，强制重新登录（文档 §3.5.3）
	if _, err := session.KickByUserIDs(ctx, []int64{id}); err != nil {
		return err
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
	defaultPassword := config.GetConfig().System.DefaultPassword
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
	// 校验用户是否存在（A0401）
	user, err := s.userRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询用户失败", err)
	}
	if user == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "用户不存在")
	}

	// 超级管理员不可禁用（防自锁），启用不受限
	if user.Username == "root" && status == 0 {
		return common.NewBizError(common.ROOT_USER_PROTECTED, "超级管理员不可禁用")
	}

	// 任何用户不可禁用自己（文档 T-UM-042）
	if operatorID := database.GetUserID(ctx); operatorID != 0 && operatorID == id {
		return common.NewBizError(common.OPERATION_NOT_ALLOW, "不可禁用自己")
	}

	if err := s.userRepo.UpdateStatusWithTime(ctx, id, status, time.Now()); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新用户状态失败", err)
	}

	// 禁用后实时踢出该用户全部在线会话（文档 §3.6.3）
	if status == 0 {
		if _, err := session.KickByUserIDs(ctx, []int64{id}); err != nil {
			return err
		}
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

// Register 用户注册流程：校验用户名、创建用户并分配 GUEST 角色
// 返回创建的用户实体及 GUEST 角色的 dataScope，供 AuthService 构建 Session
func (s *UserService) Register(ctx context.Context, username, nickname, password string) (*model.SysUser, int8, error) {
	exists, err := s.userRepo.ExistsByUsername(ctx, username)
	if err != nil {
		return nil, 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "检查用户名失败", err)
	}
	if exists {
		return nil, 0, common.NewBizError(common.DATA_EXISTS, "用户名已被注册")
	}

	guestRole, err := s.roleRepo.FindByCode(ctx, "GUEST")
	if err != nil {
		return nil, 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "查询 GUEST 角色失败", err)
	}

	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return nil, 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "密码加密失败", err)
	}

	user := &model.SysUser{
		Username: username,
		Nickname: nickname,
		Password: string(hashedPassword),
		Gender:   1,
		Status:   1,
		Deleted:  0,
	}
	// 时间截断到秒：列为 DATETIME 秒精度，直写带纳秒的 time.Now() 会被 MySQL 进位成下一刻
	user.CreatedAt = time.Now().Truncate(time.Second)
	user.UpdatedAt = time.Now().Truncate(time.Second)

	var roleIDs []int64
	var dataScope int8
	if guestRole != nil && guestRole.Status == 1 {
		roleIDs = []int64{guestRole.ID}
		dataScope = guestRole.DataScope
	}

	if err := s.userRepo.CreateWithRoles(ctx, user, roleIDs); err != nil {
		if isDuplicateKeyError(err) {
			// 并发注册竞态：两个请求都可能穿过上面的存在性预检，落库时后者撞唯一键 →
			// 必须与预检同码返回 A0501（python 捕获 IntegrityError 后同样返回 DATA_EXISTS），
			// 曾经把竞态暴露为 B0001「创建用户失败」
			return nil, 0, common.NewBizError(common.DATA_EXISTS, "用户名已被注册")
		}
		return nil, 0, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "创建用户失败", err)
	}
	return user, dataScope, nil
}

package service

import (
	"context"
	"errors"
	"regexp"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/global"
	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/repository"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
)

const (
	// ROOT_ROLE_CODE 超级管理员角色编码
	ROOT_ROLE_CODE = "ROOT"
	// ROLE_PERMS_PREFIX Redis中角色权限缓存key前缀
	ROLE_PERMS_PREFIX = "role:perms"
)

// dataScopeLabelMap 数据权限范围中文映射
var dataScopeLabelMap = map[int8]string{
	0: "全部数据",
	1: "部门及子部门数据",
	2: "本部门数据",
	3: "本人数据",
}

// RoleService 角色服务
type RoleService struct {
	roleRepo repository.IRoleRepository
	menuRepo repository.IMenuRepository
}

// NewRoleService 创建角色服务实例
func NewRoleService(roleRepo repository.IRoleRepository, menuRepo repository.IMenuRepository) *RoleService {
	return &RoleService{
		roleRepo: roleRepo,
		menuRepo: menuRepo,
	}
}

// getRepo 获取 Repository（兼容零值实例）
func (s *RoleService) getRepo() repository.IRoleRepository {
	if s.roleRepo != nil {
		return s.roleRepo
	}
	return repository.NewRoleRepository(global.DB)
}

// GetPage 角色分页列表
func (s *RoleService) GetPage(ctx context.Context, q *query.RolePageQuery) (*vo.PageResult[vo.RolePageVO], error) {
	repo := s.getRepo()

	result, err := repo.FindPage(ctx, q)
	if err != nil {
		return nil, err
	}

	// 设置数据权限范围标签
	for i := range result.List {
		if label, ok := dataScopeLabelMap[result.List[i].DataScope]; ok {
			result.List[i].DataScopeLabel = label
		}
	}

	return result, nil
}

// GetOptions 角色下拉列表
func (s *RoleService) GetOptions(ctx context.Context) ([]vo.Option, error) {
	repo := s.getRepo()
	return repo.FindOptions(ctx)
}

// Create 创建角色
func (s *RoleService) Create(ctx context.Context, form *bo.RoleFormBO) error {
	if err := s.validateRoleForm(form); err != nil {
		return err
	}

	repo := s.getRepo()

	// 检查编码是否重复
	exists, err := repo.ExistsByCode(ctx, form.Code)
	if err != nil {
		return err
	}
	if exists {
		return errors.New("角色编码已存在")
	}

	// 检查名称是否重复
	exists, err = repo.ExistsByName(ctx, form.Name)
	if err != nil {
		return err
	}
	if exists {
		return errors.New("角色名称已存在")
	}

	// 创建角色实体
	role := &model.SysRole{
		Name:      form.Name,
		Code:      form.Code,
		Sort:      form.Sort,
		Status:    form.Status,
		DataScope: form.DataScope,
		Deleted:   0,
	}
	role.CreatedAt = time.Now()
	role.UpdatedAt = time.Now()

	return repo.Create(ctx, role)
}

// Update 更新角色
func (s *RoleService) Update(ctx context.Context, id int64, form *bo.RoleFormBO) error {
	if err := s.validateRoleForm(form); err != nil {
		return err
	}

	repo := s.getRepo()

	// 查询原角色信息
	oldRole, err := repo.FindByID(ctx, id)
	if err != nil {
		return err
	}
	if oldRole == nil {
		return errors.New("角色不存在")
	}

	// 检查是否修改了角色编码
	if oldRole.Code != form.Code {
		return errors.New("角色编码不可修改")
	}

	// ROOT 角色保护
	if oldRole.Code == ROOT_ROLE_CODE {
		return errors.New("超级管理员角色不可修改")
	}

	// 检查编码是否重复（排除自身）
	exists, err := repo.ExistsByCode(ctx, form.Code, id)
	if err != nil {
		return err
	}
	if exists {
		return errors.New("角色编码已存在")
	}

	// 检查名称是否重复（排除自身）
	exists, err = repo.ExistsByName(ctx, form.Name, id)
	if err != nil {
		return err
	}
	if exists {
		return errors.New("角色名称已存在")
	}

	// 更新角色
	role := &model.SysRole{
		BaseModel: model.BaseModel{ID: id},
		Name:      form.Name,
		Code:      form.Code,
		Sort:      form.Sort,
		Status:    form.Status,
		DataScope: form.DataScope,
	}
	role.UpdatedAt = time.Now()

	if err := repo.Update(ctx, role); err != nil {
		return err
	}

	// 判断角色编码或状态是否修改，修改了则刷新权限缓存
	if oldRole.Code != form.Code || oldRole.Status != form.Status {
		if oldRole.Code != form.Code {
			s.refreshRolePermsCache(oldRole.Code, form.Code)
		} else {
			s.refreshRolePermsCache(form.Code, "")
		}
	}

	return nil
}

// GetFormData 获取角色表单数据
func (s *RoleService) GetFormData(ctx context.Context, id int64) (*bo.RoleFormBO, error) {
	repo := s.getRepo()
	return repo.GetFormData(ctx, id)
}

// UpdateStatus 更新角色状态
func (s *RoleService) UpdateStatus(ctx context.Context, id int64, status int8) error {
	if status != 0 && status != 1 {
		return errors.New("角色状态值无效，必须为0或1")
	}

	repo := s.getRepo()

	// 查询角色
	role, err := repo.FindByID(ctx, id)
	if err != nil {
		return err
	}
	if role == nil {
		return errors.New("角色不存在")
	}

	// ROOT 角色保护
	if role.Code == ROOT_ROLE_CODE {
		return errors.New("超级管理员角色不可修改状态")
	}

	// 更新状态
	if err := repo.UpdateStatus(ctx, id, status); err != nil {
		return err
	}

	// 刷新权限缓存
	s.refreshRolePermsCache(role.Code, "")

	return nil
}

// Delete 批量删除角色
func (s *RoleService) Delete(ctx context.Context, ids []int64) error {
	repo := s.getRepo()

	if len(ids) == 0 {
		return errors.New("删除的角色ID不能为空")
	}

	// 检查角色是否存在且是否有用户关联
	for _, id := range ids {
		role, err := repo.FindByID(ctx, id)
		if err != nil {
			return err
		}
		if role == nil {
			return errors.New("角色不存在")
		}

		// ROOT 角色保护
		if role.Code == ROOT_ROLE_CODE {
			return errors.New("超级管理员角色不可删除")
		}

		// 检查是否关联用户
		hasUsers, err := repo.HasUsers(ctx, id)
		if err != nil {
			return err
		}
		if hasUsers {
			return errors.New("角色【" + role.Name + "】已分配用户，请先解除关联后删除")
		}
	}

	// 批量删除
	if err := repo.Delete(ctx, ids); err != nil {
		return err
	}

	// 批量刷新权限缓存（需要查询角色编码）
	for _, id := range ids {
		role, err := repo.FindByID(ctx, id)
		if err == nil && role != nil {
			s.refreshRolePermsCache(role.Code, "")
		}
	}

	return nil
}

// GetMenuIDs 获取角色菜单ID集合
func (s *RoleService) GetMenuIDs(ctx context.Context, roleID int64) ([]int64, error) {
	repo := s.getRepo()

	// 检查角色是否存在
	role, err := repo.FindByID(ctx, roleID)
	if err != nil {
		return nil, err
	}
	if role == nil {
		return nil, errors.New("角色不存在")
	}

	return repo.GetMenuIDs(ctx, roleID)
}

// AssignMenus 分配菜单权限
func (s *RoleService) AssignMenus(ctx context.Context, roleID int64, menuIDs []int64) error {
	repo := s.getRepo()

	// 检查角色是否存在
	role, err := repo.FindByID(ctx, roleID)
	if err != nil {
		return err
	}
	if role == nil {
		return errors.New("角色不存在")
	}

	// 分配菜单
	if err := repo.AssignMenus(ctx, roleID, menuIDs); err != nil {
		return err
	}

	// 刷新权限缓存
	s.refreshRolePermsCache(role.Code, "")

	// TODO: 清除路由缓存
	// if s.menuRepo != nil {
	// 	// 清除路由缓存
	// }

	return nil
}

// ====================
// IRoleService 接口实现结束
// ====================

// ====================
// 辅助方法
// ====================

// GetMaximumDataScope 获取最大范围的数据权限
func (s *RoleService) GetMaximumDataScope(roles []string) (dataScope *int8, err error) {
	if len(roles) == 0 {
		return nil, nil
	}

	var scope int8
	// 使用 global.DB 直接查询
	err = global.DB.Model(&model.SysRole{}).
		Select("MIN(data_scope)").
		Where("code IN ?", roles).
		Where("deleted = ?", 0).
		Scan(&scope).Error

	if err != nil {
		return nil, err
	}
	return &scope, nil
}

// validateRoleForm 验证角色表单数据
func (s *RoleService) validateRoleForm(form *bo.RoleFormBO) error {
	if strings.TrimSpace(form.Name) == "" {
		return errors.New("角色名称不能为空")
	}
	if len(form.Name) < 2 || len(form.Name) > 30 {
		return errors.New("角色名称长度必须在2-30个字符之间")
	}
	if strings.TrimSpace(form.Code) == "" {
		return errors.New("角色编码不能为空")
	}
	if len(form.Code) > 50 {
		return errors.New("角色编码长度不能超过50个字符")
	}
	codePattern := regexp.MustCompile(`^[A-Z][A-Z0-9_]*$`)
	if !codePattern.MatchString(form.Code) {
		return errors.New("角色编码格式不正确，只能包含大写字母、数字和下划线")
	}
	if form.DataScope < 0 || form.DataScope > 3 {
		return errors.New("数据权限范围值无效，必须为0-3")
	}
	return nil
}

// refreshRolePermsCache 刷新角色权限缓存
// oldRoleCode: 旧角色编码（角色编码变更时使用，否则传空字符串）
// newRoleCode: 新角色编码（角色编码变更时使用，否则传当前角色编码）
func (s *RoleService) refreshRolePermsCache(oldRoleCode, newRoleCode string) {
	if global.REDIS == nil {
		return
	}

	ctx := context.Background()

	if oldRoleCode != "" && newRoleCode != "" && oldRoleCode != newRoleCode {
		global.REDIS.HDel(ctx, ROLE_PERMS_PREFIX, oldRoleCode)
		s.loadRolePermsToCache(newRoleCode)
	} else {
		roleCode := oldRoleCode
		if roleCode == "" {
			roleCode = newRoleCode
		}
		global.REDIS.HDel(ctx, ROLE_PERMS_PREFIX, roleCode)
		s.loadRolePermsToCache(roleCode)
	}
}

// loadRolePermsToCache 加载角色权限到缓存
func (s *RoleService) loadRolePermsToCache(roleCode string) {
	if global.REDIS == nil || roleCode == "" {
		return
	}
	if global.DB == nil {
		if global.LOG != nil {
			logger.Error("数据库连接未初始化，无法加载角色权限")
		}
		return
	}

	ctx := context.Background()

	var perms []string
	err := global.DB.Table("sys_menu").
		Select("DISTINCT sys_menu.perm").
		Joins("INNER JOIN sys_role_menu ON sys_menu.id = sys_role_menu.menu_id").
		Joins("INNER JOIN sys_role ON sys_role.id = sys_role_menu.role_id").
		Where("sys_role.code = ? AND sys_menu.perm IS NOT NULL AND sys_menu.perm != ''", roleCode).
		Pluck("perm", &perms).Error

	if err != nil {
		if global.LOG != nil {
			logger.Error("加载角色权限到缓存失败: " + err.Error())
		}
		return
	}

	if len(perms) > 0 {
		permsInterface := make([]interface{}, len(perms))
		for i, perm := range perms {
			permsInterface[i] = perm
		}
		global.REDIS.HSet(ctx, ROLE_PERMS_PREFIX, roleCode, permsInterface)
	}
}

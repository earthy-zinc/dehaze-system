package role

import (
	"context"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	menurepo "github.com/earthyzinc/dehaze-go/internal/repository/menu"
	rolerepo "github.com/earthyzinc/dehaze-go/internal/repository/role"
	"github.com/earthyzinc/dehaze-go/internal/service/mapper"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
)

const (
	// ROOT_ROLE_CODE 超级管理员角色编码
	ROOT_ROLE_CODE = "ROOT"
	// ROLE_PERMS_PREFIX Redis中角色权限缓存key前缀
	ROLE_PERMS_PREFIX = "role:perms:"
	// ROLE_PERMS_TTL 角色权限缓存过期时间（30分钟）
	ROLE_PERMS_TTL = 30 * time.Minute
)

// rolePermsKey 构造角色权限缓存的独立 Redis Key（逐角色独立 TTL）
func rolePermsKey(roleCode string) string {
	return ROLE_PERMS_PREFIX + roleCode
}

// dataScopeLabelMap 数据权限范围中文映射
var dataScopeLabelMap = map[int8]string{
	0: "全部数据",
	1: "部门及子部门数据",
	2: "本部门数据",
	3: "本人数据",
}

// RoleService 角色服务
type RoleService struct {
	cache    types.ICache
	roleRepo rolerepo.IRoleRepository
	menuRepo menurepo.IMenuRepository
}

// NewRoleService 创建角色服务实例
func NewRoleService(cache types.ICache, roleRepo rolerepo.IRoleRepository, menuRepo menurepo.IMenuRepository) *RoleService {
	return &RoleService{
		cache:    cache,
		roleRepo: roleRepo,
		menuRepo: menuRepo,
	}
}

// GetPage 角色分页列表
func (s *RoleService) GetPage(ctx context.Context, q *query.RolePageQuery) (*vo.PageResult[vo.RolePageVO], error) {
	readResult, err := s.roleRepo.FindPage(ctx, q)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询角色分页列表失败", err)
	}
	if readResult == nil {
		return &vo.PageResult[vo.RolePageVO]{List: []vo.RolePageVO{}, Total: 0}, nil
	}

	voList := make([]vo.RolePageVO, 0, len(readResult.List))
	for _, item := range readResult.List {
		label := item.DataScopeLabel
		if mapped, ok := dataScopeLabelMap[item.DataScope]; ok {
			label = mapped
		}
		voList = append(voList, vo.RolePageVO{
			ID:             item.ID,
			Name:           item.Name,
			Code:           item.Code,
			DataScope:      item.DataScope,
			DataScopeLabel: label,
			Status:         item.Status,
			Sort:           item.Sort,
			CreateTime:     item.CreateTime,
			UpdateTime:     item.UpdateTime,
		})
	}

	return &vo.PageResult[vo.RolePageVO]{
		List:  voList,
		Total: readResult.Total,
	}, nil
}

// GetOptions 角色下拉列表（非超级管理员不显示 ROOT 角色）
func (s *RoleService) GetOptions(ctx context.Context, isRoot bool) ([]vo.Option, error) {
	readOptions, err := s.roleRepo.FindOptions(ctx, isRoot)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询角色选项列表失败", err)
	}

	options := mapper.OptionsFromRead(readOptions)

	return options, nil
}

// Create 创建角色
func (s *RoleService) Create(ctx context.Context, form *bo.RoleFormBO) error {
	// 检查编码是否重复
	exists, err := s.roleRepo.ExistsByCode(ctx, form.Code)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "检查角色编码是否存在失败", err)
	}
	if exists {
		return common.NewBizError(common.DATA_EXISTS, "角色编码已存在")
	}

	// 检查名称是否重复
	exists, err = s.roleRepo.ExistsByName(ctx, form.Name)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "检查角色名称是否存在失败", err)
	}
	if exists {
		return common.NewBizError(common.DATA_EXISTS, "角色名称已存在")
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

	if err := s.roleRepo.Create(ctx, role); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "创建角色失败", err)
	}

	return nil
}

// Update 更新角色
func (s *RoleService) Update(ctx context.Context, id int64, form *bo.RoleFormBO) error {
	// 查询原角色信息
	oldRole, err := s.roleRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询角色信息失败", err)
	}
	if oldRole == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "角色不存在")
	}

	// 检查是否修改了角色编码
	if oldRole.Code != form.Code {
		return common.NewBizError(common.OPERATION_NOT_ALLOW, "角色编码不可修改")
	}

	// ROOT 角色保护
	if oldRole.Code == ROOT_ROLE_CODE {
		return common.NewBizError(common.OPERATION_NOT_ALLOW, "超级管理员角色不可修改")
	}

	// 检查名称是否重复（排除自身）
	exists, err := s.roleRepo.ExistsByName(ctx, form.Name, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "检查角色名称是否存在失败", err)
	}
	if exists {
		return common.NewBizError(common.DATA_EXISTS, "角色名称已存在")
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

	if err := s.roleRepo.Update(ctx, role); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新角色失败", err)
	}

	// 状态变更时刷新权限缓存
	if oldRole.Status != form.Status {
		s.refreshRolePermsCache(ctx, form.Code)
	}

	return nil
}

// GetFormData 获取角色表单数据
func (s *RoleService) GetFormData(ctx context.Context, id int64) (*bo.RoleFormBO, error) {
	form, err := s.roleRepo.GetFormData(ctx, id)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "获取角色表单数据失败", err)
	}
	if form == nil {
		return nil, nil
	}
	// 将 read model 转换为 bo
	return &bo.RoleFormBO{
		ID:        form.ID,
		Name:      form.Name,
		Code:      form.Code,
		Sort:      form.Sort,
		Status:    form.Status,
		DataScope: form.DataScope,
	}, nil
}

// UpdateStatus 更新角色状态
func (s *RoleService) UpdateStatus(ctx context.Context, id int64, status int8) error {
	if status != 0 && status != 1 {
		return common.NewBizError(common.PARAM_ERROR, "角色状态值无效，必须为0或1")
	}

	// 查询角色
	role, err := s.roleRepo.FindByID(ctx, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询角色信息失败", err)
	}
	if role == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "角色不存在")
	}

	// ROOT 角色保护
	if role.Code == ROOT_ROLE_CODE {
		return common.NewBizError(common.OPERATION_NOT_ALLOW, "超级管理员角色不可修改状态")
	}

	// 更新状态
	if err := s.roleRepo.UpdateStatus(ctx, id, status); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新角色状态失败", err)
	}

	// 刷新权限缓存
	s.refreshRolePermsCache(ctx, role.Code)

	return nil
}

// Delete 批量删除角色
func (s *RoleService) Delete(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return common.NewBizError(common.PARAM_ERROR, "删除的角色ID不能为空")
	}

	// 批量查询角色（优化N+1）
	roles, err := s.roleRepo.FindByIDs(ctx, ids)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询角色信息失败", err)
	}
	if len(roles) != len(ids) {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "部分角色不存在")
	}

	// ROOT 角色保护
	for _, role := range roles {
		if role.Code == ROOT_ROLE_CODE {
			return common.NewBizError(common.OPERATION_NOT_ALLOW, "超级管理员角色不可删除")
		}
	}

	// 批量检查是否关联用户（优化N+1）
	userMap, err := s.roleRepo.HasUsersInBatch(ctx, ids)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "检查角色是否关联用户失败", err)
	}
	for _, role := range roles {
		if userMap[role.ID] {
			return common.NewBizError(common.BUSINESS_ERROR, "角色【"+role.Name+"】已分配用户，请先解除关联后删除")
		}
	}

	// 删除前收集角色编码（修复缓存清理：删除后FindByID查不到已逻辑删除的数据）
	roleCodes := make([]string, 0, len(roles))
	for _, role := range roles {
		roleCodes = append(roleCodes, role.Code)
	}

	// 删除角色及其菜单关联（事务）
	if err := s.roleRepo.DeleteWithMenus(ctx, ids); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "删除角色失败", err)
	}

	// 批量清理权限缓存（角色已删除，直接清除缓存即可，无需重新加载权限）
	if s.cache != nil {
		keys := make([]string, 0, len(roleCodes))
		for _, code := range roleCodes {
			keys = append(keys, rolePermsKey(code))
		}
		_ = s.cache.Delete(ctx, keys...)
	}

	return nil
}

// GetMenuIDs 获取角色菜单ID集合
func (s *RoleService) GetMenuIDs(ctx context.Context, roleID int64) ([]int64, error) {
	// 检查角色是否存在
	role, err := s.roleRepo.FindByID(ctx, roleID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询角色信息失败", err)
	}
	if role == nil {
		return []int64{}, nil
	}

	menuIDs, err := s.roleRepo.GetMenuIDs(ctx, roleID)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "获取角色菜单ID集合失败", err)
	}
	if menuIDs == nil {
		return []int64{}, nil
	}
	return menuIDs, nil
}

// AssignMenus 分配菜单权限
func (s *RoleService) AssignMenus(ctx context.Context, roleID int64, menuIDs []int64) error {
	// 检查角色是否存在
	role, err := s.roleRepo.FindByID(ctx, roleID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询角色信息失败", err)
	}
	if role == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "角色不存在")
	}

	// 分配菜单
	if err := s.roleRepo.AssignMenus(ctx, roleID, menuIDs); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "分配菜单权限失败", err)
	}

	// 刷新权限缓存
	s.refreshRolePermsCache(ctx, role.Code)

	return nil
}

// ====================
// IRoleService 接口实现结束
// ====================

// ====================
// 辅助方法
// ====================

// GetMaximumDataScope 获取最大范围的数据权限
func (s *RoleService) GetMaximumDataScope(ctx context.Context, roles []string) (dataScope *int8, err error) {
	dataScope, err = s.roleRepo.GetMinimumDataScope(ctx, roles)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "获取数据权限范围失败", err)
	}
	return dataScope, nil
}

// refreshRolePermsCache 刷新角色权限缓存（删除后重新加载）
func (s *RoleService) refreshRolePermsCache(ctx context.Context, roleCode string) {
	if s.cache == nil {
		return
	}
	_ = s.cache.Delete(ctx, rolePermsKey(roleCode))
	s.loadRolePermsToCache(ctx, roleCode)
}

// loadRolePermsToCache 加载角色权限到缓存（独立 Key + 独立 TTL）
func (s *RoleService) loadRolePermsToCache(ctx context.Context, roleCode string) {
	if s.cache == nil || roleCode == "" || s.menuRepo == nil {
		return
	}

	perms, err := s.menuRepo.FindPermsByRoleCode(ctx, roleCode)
	if err != nil {
		logger.Error("加载角色权限到缓存失败: " + err.Error())
		return
	}

	if len(perms) == 0 {
		return
	}

	_ = s.cache.Set(ctx, rolePermsKey(roleCode), strings.Join(perms, ","), ROLE_PERMS_TTL)
}

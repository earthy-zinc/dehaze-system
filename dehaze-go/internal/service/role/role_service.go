package role

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	menurepo "github.com/earthyzinc/dehaze-go/internal/repository/menu"
	rolerepo "github.com/earthyzinc/dehaze-go/internal/repository/role"
	auditlogservice "github.com/earthyzinc/dehaze-go/internal/service/audit_log"
	"github.com/earthyzinc/dehaze-go/internal/service/mapper"
	rediscache "github.com/earthyzinc/dehaze-go/pkg/cache/redis"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/database"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

const (
	// ROOT_ROLE_CODE 超级管理员角色编码
	ROOT_ROLE_CODE = "ROOT"
	// ADMIN_ROLE_CODE 管理员角色编码
	ADMIN_ROLE_CODE = "ADMIN"
	// ROLE_PERMS_PREFIX Redis中角色权限缓存key前缀
	ROLE_PERMS_PREFIX = "role:perms:"
	// ROLE_PERMS_TTL 角色权限缓存过期时间（30分钟）
	ROLE_PERMS_TTL = 30 * time.Minute
	// ROLE_OPTIONS_CACHE_KEY 角色选项缓存key
	ROLE_OPTIONS_CACHE_KEY = "role:options"
	// ROLE_OPTIONS_CACHE_TTL 角色选项缓存过期时间（1小时）
	ROLE_OPTIONS_CACHE_TTL = time.Hour
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
	cache       types.ICache
	roleRepo    rolerepo.IRoleRepository
	menuRepo    menurepo.IMenuRepository
	auditLogSvc *auditlogservice.AuditLogService
}

// NewRoleService 创建角色服务实例
func NewRoleService(cache types.ICache, roleRepo rolerepo.IRoleRepository, menuRepo menurepo.IMenuRepository, auditLogSvc *auditlogservice.AuditLogService) *RoleService {
	return &RoleService{
		cache:       cache,
		roleRepo:    roleRepo,
		menuRepo:    menuRepo,
		auditLogSvc: auditLogSvc,
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
	readOptions, err := s.loadRoleOptions(ctx)
	if err != nil {
		return nil, common.WrapBizError(common.DATABASE_ERROR, "查询角色选项列表失败", err)
	}

	// 非 root 隐藏内置角色（ROOT/ADMIN，与 Python 口径一致）；缓存全量，可见性过滤按调用方视角处理
	if !isRoot {
		filtered := make([]read.Option, 0, len(readOptions))
		for _, option := range readOptions {
			if option.Code != ROOT_ROLE_CODE && option.Code != ADMIN_ROLE_CODE {
				filtered = append(filtered, option)
			}
		}
		readOptions = filtered
	}

	options := mapper.OptionsFromRead(readOptions)

	return options, nil
}

// loadRoleOptions 读取启用角色选项（role:options 缓存 TTL 1h，缓存全量含编码）
func (s *RoleService) loadRoleOptions(ctx context.Context) ([]read.Option, error) {
	if s.cache != nil {
		if cached, err := s.cache.Get(ctx, ROLE_OPTIONS_CACHE_KEY); err == nil && cached != "" {
			var options []read.Option
			if json.Unmarshal([]byte(cached), &options) == nil {
				return options, nil
			}
		}
	}

	options, err := s.roleRepo.FindOptions(ctx)
	if err != nil {
		return nil, err
	}

	if s.cache != nil {
		if data, err := json.Marshal(options); err == nil {
			_ = s.cache.Set(ctx, ROLE_OPTIONS_CACHE_KEY, string(data), ROLE_OPTIONS_CACHE_TTL)
		}
	}
	return options, nil
}

// invalidateRoleOptionsCache 角色增删改后失效选项缓存
func (s *RoleService) invalidateRoleOptionsCache(ctx context.Context) {
	if s.cache != nil {
		_ = s.cache.Delete(ctx, ROLE_OPTIONS_CACHE_KEY)
	}
}

// Create 创建角色
func (s *RoleService) Create(ctx context.Context, form *bo.RoleFormBO) error {
	if form.Code == "" {
		return common.NewBizError(common.PARAM_ERROR, "角色编码不能为空")
	}

	// 新增时 dataScope 必填（T-RM-012：数据权限未选择报"数据权限不能为空"）
	if form.DataScope == nil {
		return common.NewBizError(common.PARAM_ERROR, "数据权限不能为空")
	}

	// 检查编码是否重复（查全表含软删行）
	exists, err := s.roleRepo.ExistsByCode(ctx, form.Code)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "检查角色编码是否存在失败", err)
	}
	if exists {
		return common.NewBizError(common.DATA_EXISTS, "角色编码已存在")
	}

	// 检查名称是否重复（查全表含软删行）
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
		DataScope: *form.DataScope,
		Deleted:   0,
	}
	// 时间截断到秒：列为 DATETIME（秒精度），直写带纳秒的 time.Now() 会被 MySQL 进位成下一刻
	role.CreatedAt = time.Now().Truncate(time.Second)
	role.UpdatedAt = time.Now().Truncate(time.Second)

	if err := s.roleRepo.Create(ctx, role); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "创建角色失败", err)
	}

	s.invalidateRoleOptionsCache(ctx)

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

	// 检查名称是否重复（排除自身，查全表含软删行）
	exists, err := s.roleRepo.ExistsByName(ctx, form.Name, id)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "检查角色名称是否存在失败", err)
	}
	if exists {
		return common.NewBizError(common.DATA_EXISTS, "角色名称已存在")
	}

	// 更新角色（内置角色不可修改状态和数据权限，与 Python 端一致：忽略而非报错）
	role := &model.SysRole{
		BaseModel: model.BaseModel{ID: id},
		Name:      form.Name,
		Code:      form.Code,
		Sort:      form.Sort,
		Status:    form.Status,
	}
	// dataScope 未随请求提交时保持原值（创建时必填，编辑时可选）
	if form.DataScope != nil {
		role.DataScope = *form.DataScope
	} else {
		role.DataScope = oldRole.DataScope
	}
	if oldRole.Code == ROOT_ROLE_CODE || oldRole.Code == ADMIN_ROLE_CODE {
		role.Status = oldRole.Status
		role.DataScope = oldRole.DataScope
	}
	role.UpdatedAt = time.Now().Truncate(time.Second)

	if err := s.roleRepo.Update(ctx, role); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新角色失败", err)
	}

	// 状态变更时刷新权限缓存
	if oldRole.Status != form.Status {
		s.refreshRolePermsCache(ctx, form.Code)
	}
	s.invalidateRoleOptionsCache(ctx)

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
	label := dataScopeLabelMap[form.DataScope]
	return &bo.RoleFormBO{
		ID:             form.ID,
		Name:           form.Name,
		Code:           form.Code,
		Sort:           form.Sort,
		Status:         form.Status,
		DataScope:      &form.DataScope,
		DataScopeLabel: label,
		CreateTime:     &form.CreateTime,
		UpdateTime:     &form.UpdateTime,
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

	// 内置角色不可修改状态（ROOT/ADMIN，与 Python 端 BUILTIN_ROLE_CODES 一致）
	if role.Code == ROOT_ROLE_CODE || role.Code == ADMIN_ROLE_CODE {
		return common.NewBizError(common.OPERATION_NOT_ALLOW, "内置角色 '"+role.Code+"' 不可修改状态")
	}

	// 更新状态
	if err := s.roleRepo.UpdateStatus(ctx, id, status); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "更新角色状态失败", err)
	}

	// 刷新权限缓存
	s.refreshRolePermsCache(ctx, role.Code)
	s.invalidateRoleOptionsCache(ctx)

	// 禁用角色时踢出关联在线用户（权限传播）；启用不踢
	if status == 0 {
		s.kickRoleUserSessions(ctx, id)
	}

	if s.auditLogSvc != nil {
		s.auditLogSvc.RecordAuditAsync(ctx, database.GetUserID(ctx), "role", id, "status_change", "role", role.Status, status, database.GetIP(ctx), database.GetUserAgent(ctx))
	}

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

	// 内置角色保护（ROOT / ADMIN 禁止删除）
	for _, role := range roles {
		if role.Code == ROOT_ROLE_CODE || role.Code == ADMIN_ROLE_CODE {
			return common.NewBizError(common.OPERATION_NOT_ALLOW, "内置角色不可删除")
		}
	}

	// 批量检查是否关联用户（优化N+1）
	userMap, err := s.roleRepo.HasUsersInBatch(ctx, ids)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "检查角色是否关联用户失败", err)
	}
	for _, role := range roles {
		if userMap[role.ID] {
			return common.NewBizError(common.BUSINESS_ERROR, "该角色仍有用户关联，请先解绑")
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
	s.invalidateRoleOptionsCache(ctx)

	if s.auditLogSvc != nil {
		s.auditLogSvc.RecordAuditAsync(ctx, database.GetUserID(ctx), "role", ids, "delete", "role", roleCodes, nil, database.GetIP(ctx), database.GetUserAgent(ctx))
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

// AssignMenus 分配菜单权限（operatorPerms 为操作者权限标识集合，operatorIsRoot 标识超级管理员）
func (s *RoleService) AssignMenus(ctx context.Context, roleID int64, menuIDs []int64, operatorPerms []string, operatorIsRoot bool) error {
	// 检查角色是否存在
	role, err := s.roleRepo.FindByID(ctx, roleID)
	if err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "查询角色信息失败", err)
	}
	if role == nil {
		return common.NewBizError(common.RESOURCE_NOT_FOUND, "角色不存在")
	}

	// 菜单存在性校验：不允许分配不存在的菜单 ID（A0401）
	if len(menuIDs) > 0 {
		count, err := s.menuRepo.CountByIDs(ctx, menuIDs)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询菜单失败", err)
		}
		// 去重后比对，重复 ID 不误判为不存在
		dedup := make(map[int64]bool, len(menuIDs))
		unique := 0
		for _, id := range menuIDs {
			if !dedup[id] {
				dedup[id] = true
				unique++
			}
		}
		if count != int64(unique) {
			return common.NewBizError(common.RESOURCE_NOT_FOUND, "菜单不存在")
		}
	}

	// 权限提升防护（A0301）：操作者不能授予自己未持有的权限标识；ROOT 忽略权限判断
	if len(menuIDs) > 0 && !operatorIsRoot {
		menuPerms, err := s.menuRepo.FindPermsByMenuIDs(ctx, menuIDs)
		if err != nil {
			return common.WrapBizError(common.DATABASE_ERROR, "查询菜单权限标识失败", err)
		}
		held := make(map[string]struct{}, len(operatorPerms))
		for _, perm := range operatorPerms {
			held[perm] = struct{}{}
		}
		for _, perm := range menuPerms {
			if _, ok := held[perm]; !ok {
				return common.NewBizError(common.ACCESS_UNAUTHORIZED, "无权分配超出自身权限范围的权限标识")
			}
		}
	}

	// 分配菜单
	if err := s.roleRepo.AssignMenus(ctx, roleID, menuIDs); err != nil {
		return common.WrapBizError(common.DATABASE_ERROR, "分配菜单权限失败", err)
	}

	// 刷新权限缓存（repo 事务已提交，缓存操作在提交后执行）
	s.refreshRolePermsCache(ctx, role.Code)

	// 在线用户权限传播：踢出关联角色的在线用户
	s.kickRoleUserSessions(ctx, roleID)

	if s.auditLogSvc != nil {
		s.auditLogSvc.RecordAuditAsync(ctx, database.GetUserID(ctx), "role", roleID, "update", "role", nil, menuIDs, database.GetIP(ctx), database.GetUserAgent(ctx))
	}

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

// kickRoleUserSessions 踢出关联角色的在线用户（权限传播）。
// 不依赖多点登录索引（索引仅在 use_multi_point 开启时维护），扫描会话空间
// 按 username 匹配；超级管理员会话不可被踢出。
func (s *RoleService) kickRoleUserSessions(ctx context.Context, roleID int64) {
	client := rediscache.GetClient()
	if client == nil {
		return
	}

	usernames, err := s.roleRepo.FindUsernamesByRoleIDs(ctx, []int64{roleID})
	if err != nil {
		logger.Warn("查询角色关联用户失败", zap.Int64("roleId", roleID), zap.Error(err))
		return
	}
	if len(usernames) == 0 {
		return
	}
	target := make(map[string]struct{}, len(usernames))
	for _, name := range usernames {
		target[name] = struct{}{}
	}

	keys, err := client.Keys(ctx, common.SessionPrefix+"*").Result()
	if err != nil || len(keys) == 0 {
		return
	}
	for _, key := range keys {
		if strings.HasPrefix(key, common.SessionUserPrefix) {
			continue
		}
		raw, err := client.Get(ctx, key).Result()
		if err != nil || raw == "" {
			continue
		}
		var session struct {
			UserID      int64    `json:"userId"`
			Username    string   `json:"username"`
			Authorities []string `json:"authorities"`
		}
		if json.Unmarshal([]byte(raw), &session) != nil {
			continue
		}
		if _, ok := target[session.Username]; !ok {
			continue
		}
		if slices.Contains(session.Authorities, "ROLE_"+ROOT_ROLE_CODE) {
			continue
		}
		if err := client.Del(ctx, key).Err(); err != nil {
			logger.Warn("踢出在线会话失败", zap.String("sessionKey", key), zap.Error(err))
		}
	}
}

// loadRolePermsToCache 加载角色权限到缓存（独立 Key + 独立 TTL）
// 写入 JSON 字符串数组格式（如 ["perm1","perm2"] 或 []），与 Java/Python 端保持一致
func (s *RoleService) loadRolePermsToCache(ctx context.Context, roleCode string) {
	if s.cache == nil || roleCode == "" || s.menuRepo == nil {
		return
	}

	perms, err := s.menuRepo.FindPermsByRoleCode(ctx, roleCode)
	if err != nil {
		logger.Error("加载角色权限到缓存失败: " + err.Error())
		return
	}

	if perms == nil {
		perms = []string{}
	}

	data, err := json.Marshal(perms)
	if err != nil {
		logger.Error("序列化角色权限失败: " + err.Error())
		return
	}
	_ = s.cache.Set(ctx, rolePermsKey(roleCode), string(data), ROLE_PERMS_TTL)
}

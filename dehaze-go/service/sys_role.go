package service

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"time"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/model"
	"github.com/earthyzinc/dehaze-go/model/bo"
	"github.com/earthyzinc/dehaze-go/model/query"
	"github.com/earthyzinc/dehaze-go/model/vo"
	"gorm.io/gorm"
)

const (
	// ROOT_ROLE_CODE 超级管理员角色编码
	ROOT_ROLE_CODE = "ROOT"
	// ROLE_PERMS_PREFIX Redis中角色权限缓存key前缀
	ROLE_PERMS_PREFIX = "role:perms"
)

type RoleService struct{}

// GetRolePage 角色分页列表
func (roleService *RoleService) GetRolePage(queryParams query.RolePageQuery) (result vo.PageResult[vo.RolePageVO], err error) {
	// 检查全局数据库连接
	if global.DB == nil {
		return result, errors.New("数据库连接未初始化")
	}

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
	db := global.DB.Model(&model.SysRole{}).
		Where("deleted = ?", 0)

	// TODO: 添加ROOT角色过滤 - 需要获取当前用户角色判断是否为超级管理员
	// 非超级管理员不显示ROOT角色
	// isRoot := checkIfCurrentUserIsRoot() // 需要从context获取当前用户信息
	// if !isRoot {
	// 	db = db.Where("code != ?", ROOT_ROLE_CODE)
	// }

	// 添加查询条件
	if queryParams.Keywords != "" {
		keyword := "%" + queryParams.Keywords + "%"
		db = db.Where("name LIKE ? OR code LIKE ?", keyword, keyword)
	}

	// 查询总数
	var total int64
	err = db.Count(&total).Error
	if err != nil {
		return result, err
	}

	// 分页查询
	var roles []model.SysRole
	err = db.Offset((pageNum - 1) * pageSize).Limit(pageSize).Find(&roles).Error
	if err != nil {
		return result, err
	}

	// 转换为VO
	var rolePageVOs []vo.RolePageVO
	for _, role := range roles {
		rolePageVO := vo.RolePageVO{
			ID:         role.ID,
			Name:       role.Name,
			Code:       role.Code,
			Status:     int(role.Status),
			Sort:       role.Sort,
			CreateTime: role.CreatedAt,
			UpdateTime: role.UpdatedAt,
		}
		rolePageVOs = append(rolePageVOs, rolePageVO)
	}

	// 构造分页结果
	result.List = rolePageVOs
	result.Total = total
	result.PageNum = pageNum
	result.PageSize = pageSize

	return result, nil
}

// ListRoleOptions 角色下拉列表
func (roleService *RoleService) ListRoleOptions() (options []vo.Option, err error) {
	// 检查全局数据库连接
	if global.DB == nil {
		return options, errors.New("数据库连接未初始化")
	}

	// 查询数据
	db := global.DB.Model(&model.SysRole{}).
		Where("deleted = ?", 0)

	// TODO: 添加ROOT角色过滤 - 需要获取当前用户角色判断是否为超级管理员
	// 非超级管理员不显示ROOT角色
	// isRoot := checkIfCurrentUserIsRoot()
	// if !isRoot {
	// 	db = db.Where("code != ?", ROOT_ROLE_CODE)
	// }

	var roles []model.SysRole
	err = db.Select("id, name").
		Order("sort ASC").
		Find(&roles).Error

	if err != nil {
		return options, err
	}

	// 转换为Option
	for _, role := range roles {
		option := vo.Option{
			Value: role.ID,
			Label: role.Name,
		}
		options = append(options, option)
	}

	return options, nil
}

// SaveRole 保存角色
func (roleService *RoleService) SaveRole(roleFormBO bo.RoleFormBO) (err error) {
	// 检查全局数据库连接
	if global.DB == nil {
		return errors.New("数据库连接未初始化")
	}

	// 输入参数验证
	if strings.TrimSpace(roleFormBO.Name) == "" {
		return errors.New("角色名称不能为空")
	}
	if len(roleFormBO.Name) > 50 {
		return errors.New("角色名称长度不能超过50个字符")
	}
	if strings.TrimSpace(roleFormBO.Code) == "" {
		return errors.New("角色编码不能为空")
	}
	if len(roleFormBO.Code) > 50 {
		return errors.New("角色编码长度不能超过50个字符")
	}
	if roleFormBO.Status != 0 && roleFormBO.Status != 1 {
		return errors.New("角色状态值无效，必须为0或1")
	}

	var roleId int64
	if roleFormBO.ID != nil {
		roleId = *roleFormBO.ID
	}

	// 编辑角色时，判断角色是否存在
	var oldRole *model.SysRole
	if roleId != 0 {
		oldRole = &model.SysRole{}
		err = global.DB.Where("id = ?", roleId).First(oldRole).Error
		if err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return errors.New("角色不存在")
			}
			return err
		}
	}

	// 检查角色名称或编码是否已存在
	var count int64
	err = global.DB.Model(&model.SysRole{}).
		Where("code = ? OR name = ?", roleFormBO.Code, roleFormBO.Name).
		Where("id != ?", roleId).
		Where("deleted = ?", 0).
		Count(&count).Error

	if err != nil {
		return err
	}

	if count > 0 {
		return errors.New("角色名称或角色编码已存在，请修改后重试！")
	}

	// 创建或更新角色
	role := model.SysRole{
		Name:      roleFormBO.Name,
		Code:      roleFormBO.Code,
		Sort:      roleFormBO.Sort,
		Status:    roleFormBO.Status,
		DataScope: roleFormBO.DataScope,
		Deleted:   0,
	}

	// 开启事务
	tx := global.DB.Begin()
	if tx.Error != nil {
		return tx.Error
	}

	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	if roleId != 0 {
		// 更新角色
		role.ID = roleId
		role.UpdatedAt = time.Now()
		// 使用Select明确指定要更新的字段，包括可能为零值的status字段
		err = tx.Model(&model.SysRole{}).Where("id = ?", roleId).
			Select("name", "code", "sort", "status", "data_scope", "updated_at").
			Updates(role).Error
		if err != nil {
			tx.Rollback()
			return err
		}

		// 判断角色编码或状态是否修改，修改了则刷新权限缓存
		if oldRole.Code != roleFormBO.Code || oldRole.Status != roleFormBO.Status {
			// 刷新权限缓存
			if oldRole.Code != roleFormBO.Code {
				// 角色编码变更，需要删除旧缓存并添加新缓存
				roleService.refreshRolePermsCache(oldRole.Code, roleFormBO.Code)
			} else {
				// 仅状态变更，刷新当前角色缓存
				roleService.refreshRolePermsCache(roleFormBO.Code, "")
			}
		}
	} else {
		// 创建角色
		role.CreatedAt = time.Now()
		role.UpdatedAt = time.Now()
		err = tx.Create(&role).Error
		if err != nil {
			tx.Rollback()
			return err
		}
	}

	// 提交事务
	err = tx.Commit().Error
	if err != nil {
		return err
	}

	return nil
}

// GetRoleForm 获取角色表单数据
func (roleService *RoleService) GetRoleForm(roleId int64) (roleFormBO bo.RoleFormBO, err error) {
	// 检查全局数据库连接
	if global.DB == nil {
		return roleFormBO, errors.New("数据库连接未初始化")
	}

	var role model.SysRole
	err = global.DB.Where("id = ?", roleId).First(&role).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return roleFormBO, errors.New("角色不存在")
		}
		return roleFormBO, err
	}

	roleFormBO = bo.RoleFormBO{
		ID:        &role.ID,
		Name:      role.Name,
		Code:      role.Code,
		Sort:      role.Sort,
		Status:    role.Status,
		DataScope: role.DataScope,
	}

	return roleFormBO, nil
}

// UpdateRoleStatus 修改角色状态
func (roleService *RoleService) UpdateRoleStatus(roleId int64, status int8) (err error) {
	// 检查全局数据库连接
	if global.DB == nil {
		return errors.New("数据库连接未初始化")
	}

	// 输入参数验证
	if status != 0 && status != 1 {
		return errors.New("角色状态值无效，必须为0或1")
	}

	// 检查角色是否存在
	var role model.SysRole
	err = global.DB.Where("id = ?", roleId).First(&role).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return errors.New("角色不存在")
		}
		return err
	}

	// 开启事务
	tx := global.DB.Begin()
	if tx.Error != nil {
		return tx.Error
	}

	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	// 更新状态
	err = tx.Model(&model.SysRole{}).
		Where("id = ?", roleId).
		Updates(map[string]interface{}{
			"status":      status,
			"update_time": time.Now(),
		}).Error

	if err != nil {
		tx.Rollback()
		return err
	}

	// 提交事务
	err = tx.Commit().Error
	if err != nil {
		return err
	}

	// 刷新角色的权限缓存
	roleService.refreshRolePermsCache(role.Code, "")

	return nil
}

// DeleteRoles 批量删除角色
func (roleService *RoleService) DeleteRoles(ids string) (err error) {
	// 检查全局数据库连接
	if global.DB == nil {
		return errors.New("数据库连接未初始化")
	}

	if ids == "" {
		return errors.New("删除的角色ID不能为空")
	}

	// 解析ID列表
	idStrings := strings.Split(ids, ",")
	var idList []int64
	for _, idStr := range idStrings {
		id, err := strconv.ParseInt(idStr, 10, 64)
		if err != nil {
			return errors.New("角色ID格式不正确")
		}
		idList = append(idList, id)
	}

	// 开启事务
	tx := global.DB.Begin()
	if tx.Error != nil {
		return tx.Error
	}

	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	// 检查角色是否存在
	var roles []model.SysRole
	err = tx.Where("id IN ? AND deleted = ?", idList, 0).Find(&roles).Error
	if err != nil {
		tx.Rollback()
		return err
	}

	if len(roles) != len(idList) {
		tx.Rollback()
		return errors.New("部分角色不存在")
	}

	// 批量检查角色是否被用户关联（优化N+1查询）
	roleIds := make([]int64, len(roles))
	roleMap := make(map[int64]*model.SysRole)
	for i, role := range roles {
		roleIds[i] = role.ID
		roleMap[role.ID] = &roles[i]
	}

	// 一次性查询所有角色的用户关联数量
	type RoleUserCount struct {
		RoleID int64
		Count  int64
	}
	var roleUserCounts []RoleUserCount
	err = tx.Model(&model.SysUserRole{}).
		Select("role_id, COUNT(*) as count").
		Where("role_id IN ?", roleIds).
		Group("role_id").
		Find(&roleUserCounts).Error

	if err != nil {
		tx.Rollback()
		return err
	}

	// 检查是否有角色被用户关联
	for _, ruc := range roleUserCounts {
		if ruc.Count > 0 {
			role := roleMap[ruc.RoleID]
			tx.Rollback()
			return errors.New("角色【" + role.Name + "】已分配用户，请先解除关联后删除")
		}
	}

	// 批量逻辑删除角色
	err = tx.Model(&model.SysRole{}).
		Where("id IN ?", roleIds).
		Updates(map[string]interface{}{
			"deleted":     1,
			"update_time": time.Now(),
		}).Error

	if err != nil {
		tx.Rollback()
		return err
	}

	// 提交事务后刷新权限缓存
	err = tx.Commit().Error
	if err != nil {
		return err
	}

	// 批量刷新角色的权限缓存
	for _, role := range roles {
		roleService.refreshRolePermsCache(role.Code, "")
	}

	return nil
}

// GetRoleMenuIds 获取角色的菜单ID集合
func (roleService *RoleService) GetRoleMenuIds(roleId int64) (menuIds []int64, err error) {
	// 检查全局数据库连接
	if global.DB == nil {
		return menuIds, errors.New("数据库连接未初始化")
	}

	// 检查角色是否存在
	var role model.SysRole
	err = global.DB.Where("id = ?", roleId).First(&role).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return menuIds, errors.New("角色不存在")
		}
		return menuIds, err
	}

	// 查询角色菜单ID集合
	err = global.DB.Model(&model.SysRoleMenu{}).
		Where("role_id = ?", roleId).
		Pluck("menu_id", &menuIds).Error

	return menuIds, err
}

// AssignMenusToRole 修改角色的资源权限
func (roleService *RoleService) AssignMenusToRole(roleId int64, menuIds []int64) (err error) {
	// 检查全局数据库连接
	if global.DB == nil {
		return errors.New("数据库连接未初始化")
	}

	// 检查角色是否存在
	var role model.SysRole
	err = global.DB.Where("id = ?", roleId).First(&role).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return errors.New("角色不存在")
		}
		return err
	}

	// 开启事务
	tx := global.DB.Begin()
	if tx.Error != nil {
		return tx.Error
	}

	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
		}
	}()

	// 删除角色菜单
	err = tx.Where("role_id = ?", roleId).Delete(&model.SysRoleMenu{}).Error
	if err != nil {
		tx.Rollback()
		return err
	}

	// 新增角色菜单
	if len(menuIds) > 0 {
		var roleMenus []map[string]interface{}
		for _, menuId := range menuIds {
			roleMenu := map[string]interface{}{
				"role_id": roleId,
				"menu_id": menuId,
			}
			roleMenus = append(roleMenus, roleMenu)
		}
		if len(roleMenus) > 0 {
			err = tx.Table("sys_role_menu").CreateInBatches(roleMenus, len(roleMenus)).Error
			if err != nil {
				tx.Rollback()
				return err
			}
		}
	}

	// 提交事务
	err = tx.Commit().Error
	if err != nil {
		return err
	}

	// 刷新角色的权限缓存
	roleService.refreshRolePermsCache(role.Code, "")

	// TODO: 清除路由缓存
	// 对应Java的 @CacheEvict(cacheNames = "menu", key = "'routes'")
	// if global.REDIS != nil {
	// 	global.REDIS.Del(context.Background(), "menu:routes")
	// }

	return nil
}

// GetMaximumDataScope 获取最大范围的数据权限
func (roleService *RoleService) GetMaximumDataScope(roles []string) (dataScope *int8, err error) {
	// 检查全局数据库连接
	if global.DB == nil {
		return nil, errors.New("数据库连接未初始化")
	}

	if len(roles) == 0 {
		return nil, nil
	}

	err = global.DB.Model(&model.SysRole{}).
		Select("MIN(data_scope)").
		Where("code IN ?", roles).
		Where("deleted = ?", 0).
		Scan(&dataScope).Error

	return dataScope, err
}

// refreshRolePermsCache 刷新角色权限缓存
// oldRoleCode: 旧角色编码（角色编码变更时使用，否则传空字符串）
// newRoleCode: 新角色编码（角色编码变更时使用，否则传当前角色编码）
func (roleService *RoleService) refreshRolePermsCache(oldRoleCode, newRoleCode string) {
	if global.REDIS == nil {
		// Redis未初始化，跳过缓存刷新
		return
	}

	ctx := context.Background()

	// 如果是角色编码变更，删除旧缓存
	if oldRoleCode != "" && newRoleCode != "" && oldRoleCode != newRoleCode {
		global.REDIS.HDel(ctx, ROLE_PERMS_PREFIX, oldRoleCode)
		// 重新加载新角色的权限到缓存
		roleService.loadRolePermsToCache(newRoleCode)
	} else {
		// 仅刷新当前角色缓存
		roleCode := oldRoleCode
		if roleCode == "" {
			roleCode = newRoleCode
		}
		global.REDIS.HDel(ctx, ROLE_PERMS_PREFIX, roleCode)
		roleService.loadRolePermsToCache(roleCode)
	}
}

// loadRolePermsToCache 加载角色权限到缓存
func (roleService *RoleService) loadRolePermsToCache(roleCode string) {
	if global.REDIS == nil || roleCode == "" {
		return
	}
	if global.DB == nil {
		if global.LOG != nil {
			global.LOG.Error("数据库连接未初始化，无法加载角色权限")
		}
		return
	}

	ctx := context.Background()

	// 查询角色的所有权限（菜单的权限标识）
	var perms []string
	err := global.DB.Table("sys_menu").
		Select("DISTINCT sys_menu.perm").
		Joins("INNER JOIN sys_role_menu ON sys_menu.id = sys_role_menu.menu_id").
		Joins("INNER JOIN sys_role ON sys_role.id = sys_role_menu.role_id").
		Where("sys_role.code = ? AND sys_menu.perm IS NOT NULL AND sys_menu.perm != ''", roleCode).
		Pluck("perm", &perms).Error

	if err != nil {
		// 记录错误但不中断流程
		if global.LOG != nil {
			global.LOG.Error("加载角色权限到缓存失败: " + err.Error())
		}
		return
	}

	// 将权限列表存入Redis
	if len(perms) > 0 {
		// 将[]string转换为[]interface{}
		permsInterface := make([]interface{}, len(perms))
		for i, perm := range perms {
			permsInterface[i] = perm
		}
		global.REDIS.HSet(ctx, ROLE_PERMS_PREFIX, roleCode, permsInterface)
	}
}

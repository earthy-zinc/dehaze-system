package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollectionUtil;
import cn.hutool.core.util.ObjectUtil;
import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.annotation.AuditLog;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.IdUtils;
import com.pei.dehaze.converter.RoleConverter;
import com.pei.dehaze.mapper.SysRoleMapper;
import com.pei.dehaze.model.entity.SysRole;
import com.pei.dehaze.model.entity.SysRoleMenu;
import com.pei.dehaze.model.form.RoleForm;
import com.pei.dehaze.model.query.RolePageQuery;
import com.pei.dehaze.model.vo.RolePageVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.SysRoleMenuService;
import com.pei.dehaze.service.SysRoleService;
import com.pei.dehaze.service.SysUserRoleService;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Set;

/**
 * 角色业务实现类
 *
 * @author earthyzinc
 * @since 2022/6/3
 */
@Service
@RequiredArgsConstructor
public class SysRoleServiceImpl extends ServiceImpl<SysRoleMapper, SysRole> implements SysRoleService {

    private final SysRoleMenuService roleMenuService;
    private final SysUserRoleService userRoleService;
    private final RoleConverter roleConverter;
    private final StringRedisTemplate stringRedisTemplate;

    /**
     * 内置角色编码集合（禁止删除）
     */
    private static final Set<String> BUILTIN_ROLE_CODES = Set.of(
            SystemConstants.ROOT_ROLE_CODE, "ADMIN"
    );

    /**
     * 角色分页列表
     *
     * @param queryParams 角色查询参数
     * @return {@link Page<RolePageVO>} – 角色分页列表
     */
    @Override
    public Page<RolePageVO> getRolePage(RolePageQuery queryParams) {
        int pageNum = queryParams.getPageNum();
        int pageSize = queryParams.getPageSize();
        String keywords = queryParams.getKeywords();

        Page<SysRole> rolePage = this.page(new Page<>(pageNum, pageSize),
                excludeRootRoleForNonRoot(new LambdaQueryWrapper<SysRole>()
                        .and(StrUtil.isNotBlank(keywords),
                                wrapper ->
                                        wrapper.like(StrUtil.isNotBlank(keywords), SysRole::getName, keywords)
                                                .or()
                                                .like(StrUtil.isNotBlank(keywords), SysRole::getCode, keywords)
                        ))
        );

        return roleConverter.entity2Page(rolePage);
    }

    /**
     * 角色下拉列表
     *
     * @return {@link List<Option>} – 角色下拉列表
     */
    @Override
    public List<Option<Long>> listRoleOptions() {
        List<SysRole> roleList = this.list(excludeRootRoleForNonRoot(new LambdaQueryWrapper<SysRole>()
                .select(SysRole::getId, SysRole::getName)
                .orderByAsc(SysRole::getSort)
        ));

        return roleConverter.entities2Options(roleList);
    }

    /**
     * 保存角色（新增或修改）
     *
     * @param roleForm 角色表单数据
     * @return {@link Boolean}
     */
    @Override
    @CacheEvict(value = "user:auth", allEntries = true)
    public boolean saveRole(RoleForm roleForm) {
        Long roleId = roleForm.getId();

        // 编辑模式：校验角色存在性 + code不可改
        SysRole oldRole = null;
        if (roleId != null) {
            oldRole = this.getById(roleId);
            if (oldRole == null) {
                throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "角色不存在");
            }
            if (!StrUtil.equals(oldRole.getCode(), roleForm.getCode())) {
                throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "角色编码不可修改");
            }
        }

        // code非空校验（schema已NOT NULL，此处双重保障）
        String roleCode = roleForm.getCode();
        if (StrUtil.isBlank(roleCode)) {
            throw new BusinessException(ResultCode.PARAM_IS_NULL, "角色编码不能为空");
        }

        // 查重：绕过@TableLogic，查全表（含软删行）
        // 命中名称或编码 → 报"已被历史记录占用"
        long count = getBaseMapper().countByNameOrCodeAll(roleForm.getName(), roleCode, roleId);
        if (count > 0) {
            throw new BusinessException(ResultCode.DATA_EXISTS,
                    "角色名称或角色编码已被历史记录占用，无法重复创建");
        }

        SysRole role = roleConverter.form2Entity(roleForm);
        boolean result = this.saveOrUpdate(role);
        if (result) {
            if (oldRole != null
                    && (!StrUtil.equals(oldRole.getCode(), roleCode)
                    || !ObjectUtil.equals(oldRole.getStatus(), roleForm.getStatus()))) {
                roleMenuService.refreshRolePermsCache(oldRole.getCode(), roleCode);
            }
        }
        return result;
    }

    /**
     * 获取角色表单数据
     *
     * @param roleId 角色ID
     * @return {@link RoleForm} – 角色表单数据
     */
    @Override
    public RoleForm getRoleForm(Long roleId) {
        SysRole entity = this.getById(roleId);
        return roleConverter.entity2Form(entity);
    }

    /**
     * 修改角色状态
     *
     * @param roleId 角色ID
     * @param status 角色状态(1:启用；0:禁用)
     * @return {@link Boolean}
     */
    @Override
    @AuditLog(module = "role", action = "status_change", targetType = "role", targetIdSpel = "#roleId", afterSpel = "#status")
    public boolean updateRoleStatus(Long roleId, Integer status) {

        SysRole role = this.getById(roleId);
        if (role == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "角色不存在");
        }

        role.setStatus(status);
        boolean result = this.updateById(role);
        if (result) {
            roleMenuService.refreshRolePermsCache(role.getCode());
        }
        return result;
    }

    /**
     * 批量删除角色
     * <p>前置校验：</p>
     * <ul>
     *   <li>内置角色（ROOT/ADMIN）禁止删除</li>
     *   <li>仍有关联用户的角色不允许删除，需先解绑</li>
     * </ul>
     * 删除后同步物理清理 sys_role_menu / sys_user_role 关联记录。
     *
     * @param ids 角色ID，多个使用英文逗号(,)分割
     * @return {@link Boolean}
     */
    @Override
    @AuditLog(module = "role", action = "delete", targetType = "role", targetIdSpel = "#ids")
    @Transactional(rollbackFor = Exception.class)
    public boolean deleteRoles(String ids) {
        if (StrUtil.isBlank(ids)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "删除的角色ID不能为空");
        }
        List<Long> roleIds = IdUtils.parseIdList(ids);

        // 批量查询角色
        List<SysRole> roles = this.listByIds(roleIds);
        if (roles.size() != roleIds.size()) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "角色不存在");
        }

        // 1) 内置角色禁止删除
        for (SysRole role : roles) {
            if (BUILTIN_ROLE_CODES.contains(role.getCode())) {
                throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW,
                        "内置角色【" + role.getName() + "】禁止删除");
            }
        }

        // 2) 仍有关联用户的角色不允许删除
        for (SysRole role : roles) {
            if (userRoleService.hasAssignedUsers(role.getId())) {
                throw new BusinessException(ResultCode.BUSINESS_ERROR,
                        "角色【" + role.getName() + "】仍有用户关联，请先解绑");
            }
        }

        // 3) 先物理清理关联记录（避免权限缓存残留）
        roleMenuService.remove(new LambdaQueryWrapper<SysRoleMenu>()
                .in(SysRoleMenu::getRoleId, roleIds));
        userRoleService.remove(new LambdaQueryWrapper<com.pei.dehaze.model.entity.SysUserRole>()
                .in(com.pei.dehaze.model.entity.SysUserRole::getRoleId, roleIds));

        // 4) 软删角色本身
        boolean deleteResult = this.removeByIds(roleIds);
        if (deleteResult) {
            for (SysRole role : roles) {
                roleMenuService.refreshRolePermsCache(role.getCode());
            }
        }
        return deleteResult;
    }

    /**
     * 获取角色的菜单ID集合
     *
     * @param roleId 角色ID
     * @return 菜单ID集合(包括按钮权限ID)
     */
    @Override
    public List<Long> getRoleMenuIds(Long roleId) {
        return roleMenuService.listMenuIdsByRoleId(roleId);
    }

    /**
     * 修改角色的资源权限
     *
     * @param roleId  角色ID
     * @param menuIds 菜单ID集合
     * @return {@link Boolean}
     */
    @Override
    @Transactional(rollbackFor = Exception.class)
    @AuditLog(module = "role", action = "update", targetType = "role", targetIdSpel = "#roleId", afterSpel = "#menuIds")
    public boolean assignMenusToRole(Long roleId, List<Long> menuIds) {
        SysRole role = this.getById(roleId);
        if (role == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "角色不存在");
        }
        roleMenuService.remove(
                new LambdaQueryWrapper<SysRoleMenu>()
                        .eq(SysRoleMenu::getRoleId, roleId)
        );
        if (CollectionUtil.isNotEmpty(menuIds)) {
            List<SysRoleMenu> roleMenus = menuIds
                    .stream()
                    .map(menuId -> new SysRoleMenu(roleId, menuId))
                    .toList();
            roleMenuService.saveBatch(roleMenus);
        }

        stringRedisTemplate.delete("menu:routes");
        roleMenuService.refreshRolePermsCache(role.getCode());

        return true;
    }

    /**
     * 获取最大范围的数据权限
     *
     * @param roles 角色编码集合
     * @return {@link Integer} – 数据权限范围
     */
    @Override
    public Integer getMaximumDataScope(Set<String> roles) {
        return this.baseMapper.getMaximumDataScope(roles);
    }

    /**
     * 为非超级管理员过滤掉超级管理员角色（超级管理员可见全部）
     */
    private LambdaQueryWrapper<SysRole> excludeRootRoleForNonRoot(LambdaQueryWrapper<SysRole> wrapper) {
        return wrapper.ne(!SecurityUtils.isRoot(), SysRole::getCode, SystemConstants.ROOT_ROLE_CODE);
    }

}

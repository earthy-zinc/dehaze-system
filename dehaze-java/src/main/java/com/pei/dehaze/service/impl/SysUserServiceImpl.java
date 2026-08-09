package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.collection.CollectionUtil;
import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.annotation.AuditLog;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.DateUtils;
import com.pei.dehaze.common.util.IdUtils;
import com.pei.dehaze.converter.UserConverter;
import com.pei.dehaze.mapper.SysUserMapper;
import com.pei.dehaze.model.read.UserRead;
import com.pei.dehaze.model.dto.UserAuthInfo;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.form.UserForm;
import com.pei.dehaze.model.query.UserPageQuery;
import com.pei.dehaze.model.vo.UserInfoVO;
import com.pei.dehaze.model.vo.UserPageVO;
import com.pei.dehaze.security.service.PermissionService;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.SysMenuService;
import com.pei.dehaze.service.SysRoleService;
import com.pei.dehaze.service.SysUserRoleService;
import com.pei.dehaze.service.SysUserService;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * 用户业务实现类
 *
 * @author earthyzinc
 * @since 2022/1/14
 */
@Service
@RequiredArgsConstructor
public class SysUserServiceImpl extends ServiceImpl<SysUserMapper, SysUser> implements SysUserService {

    private final PasswordEncoder passwordEncoder;

    private final SysUserRoleService userRoleService;

    private final UserConverter userConverter;

    private final SysMenuService menuService;

    private final SysRoleService roleService;

    private final PermissionService permissionService;

    /**
     * 获取用户分页列表
     *
     * @param queryParams 查询参数
     * @return {@link IPage<UserPageVO>} 用户分页列表
     */
    @Override
    public IPage<UserPageVO> listPagedUsers(UserPageQuery queryParams) {

        // 参数构建
        int pageNum = queryParams.getPageNum();
        int pageSize = queryParams.getPageSize();
        Page<UserRead> page = new Page<>(pageNum, pageSize);

        // 格式化为数据库日期格式，避免日期比较使用格式化函数导致索引失效
        DateUtils.toDatabaseFormat(queryParams, "startTime", "endTime");

        // 查询数据
        Page<UserRead> userPage = this.baseMapper.listPagedUsers(page, queryParams);

        // 实体转换
        return userConverter.read2PageVo(userPage);
    }

    /**
     * 获取用户表单数据
     *
     * @param userId 用户ID
     * @return
     */
    @Override
    public UserForm getUserFormData(Long userId) {
        return this.baseMapper.getUserFormData(userId);
    }

    /**
     * 新增用户 — 注册查重必须查全表（含软删行），命中报"该用户名不可用"。
     * MyBatis-Plus @TableLogic 会自动追加 deleted=0，此处必须用原生 SQL 绕过。
     *
     * @param userForm 用户表单对象
     * @return
     */
    @Override
    public boolean saveUser(UserForm userForm) {

        String username = userForm.getUsername();

        long count = this.baseMapper.countByUsernameAllDeleted(username);
        if (count > 0) {
            throw new BusinessException(ResultCode.DATA_EXISTS, "该用户名不可用");
        }

        // 实体转换 form->entity
        SysUser entity = userConverter.form2Entity(userForm);

        // 设置默认加密密码
        String defaultEncryptPwd = passwordEncoder.encode(SystemConstants.DEFAULT_PASSWORD);
        entity.setPassword(defaultEncryptPwd);

        // 新增用户
        boolean result = this.save(entity);

        if (result) {
            // 保存用户角色
            userRoleService.saveUserRoles(entity.getId(), userForm.getRoleIds());
        }
        return result;
    }

    /**
     * 更新用户 — 改名查重必须查全表（含软删行），命中报"该用户名不可用"。
     *
     * @param userId   用户ID
     * @param userForm 用户表单对象
     * @return
     */
    @Override
    @Transactional(rollbackFor = Exception.class)
    @CacheEvict(value = "user:auth", allEntries = true)
    public boolean updateUser(Long userId, UserForm userForm) {

        String username = userForm.getUsername();

        long count = this.baseMapper.countByUsernameAllDeleted(username);
        if (count > 0 && !isCurrentUser(username, userId)) {
            throw new BusinessException(ResultCode.DATA_EXISTS, "该用户名不可用");
        }

        // form -> entity
        SysUser entity = userConverter.form2Entity(userForm);
        // 设置用户ID
        entity.setId(userId);

        // 修改用户
        boolean result = this.updateById(entity);

        if (result) {
            // 保存用户角色
            userRoleService.saveUserRoles(entity.getId(), userForm.getRoleIds());
        }
        return result;
    }

    /**
     * 删除用户
     *
     * @param idsStr 用户ID，多个以英文逗号(,)分割
     * @return true|false
     */
    @Override
    @AuditLog(module = "user", action = "delete", targetType = "user", targetIdSpel = "#idsStr")
    public boolean deleteUsers(String idsStr) {
        if (StrUtil.isBlank(idsStr)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "删除的用户数据为空");
        }
        // 逻辑删除
        List<Long> ids = IdUtils.parseIdList(idsStr);
        return this.removeByIds(ids);

    }

    /**
     * 修改用户密码
     *
     * @param userId   用户ID
     * @param password 用户密码
     * @return true|false
     */
    @Override
    @AuditLog(module = "user", action = "password_change", targetType = "user", targetIdSpel = "#userId")
    public boolean updatePassword(Long userId, String password) {
        if (StrUtil.isBlank(password)) {
            throw new BusinessException(ResultCode.PARAM_IS_NULL);
        }
        Long currentUserId = SecurityUtils.getUserId();
        return this.update(new LambdaUpdateWrapper<SysUser>()
                .eq(SysUser::getId, userId)
                .set(SysUser::getPassword, passwordEncoder.encode(password))
                .set(SysUser::getUpdateBy, currentUserId)
        );
    }

    /**
     * 根据用户名获取认证信息（带缓存，TTL 由 CacheManager 统一管理）
     * <p>
     * 缓存对象为 {@link UserAuthInfo}（纯 POJO，仅含 Long/String/Integer/Set&lt;String&gt;，
     * Jackson 可正确往返序列化）。禁止缓存 SysUserDetails —— 其 authorities 字段类型
     * SimpleGrantedAuthority 无法被 Jackson 反序列化，会导致登录失败误判为密码错误。
     * 用户信息变更通过本类 update* 方法上的 @CacheEvict 清除。
     *
     * @param username 用户名
     * @return 用户认证信息 {@link UserAuthInfo}
     */
    @Override
    @Cacheable(value = "user:auth", key = "#username")
    public UserAuthInfo getUserAuthInfo(String username) {
        UserAuthInfo userAuthInfo = this.baseMapper.getUserAuthInfo(username);
        if (userAuthInfo == null) {
            throw new BusinessException(ResultCode.USERNAME_OR_PASSWORD_ERROR);
        }

        Set<String> roles = userAuthInfo.getRoles();
        if (CollectionUtil.isNotEmpty(roles)) {
            Set<String> perms = menuService.listRolePerms(roles);
            userAuthInfo.setPerms(perms);
        }

        // 获取最大范围的数据权限
        Integer dataScope = roleService.getMaximumDataScope(roles);
        userAuthInfo.setDataScope(dataScope);

        return userAuthInfo;
    }


    /**
     * 获取登录用户信息
     *
     * @return {@link UserInfoVO}   用户信息
     */
    @Override
    public UserInfoVO getCurrentUserInfo() {

        String username = Objects.requireNonNull(SecurityUtils.getUser()).getUsername(); // 登录用户名

        // 获取登录用户基础信息
        SysUser user = this.getOne(new LambdaQueryWrapper<SysUser>()
                .eq(SysUser::getUsername, username)
                .select(
                        SysUser::getId,
                        SysUser::getUsername,
                        SysUser::getNickname,
                        SysUser::getAvatar,
                        SysUser::getCreateTime
                )
        );
        // entity->VO
        UserInfoVO userInfoVO = userConverter.toUserInfoVo(user);

        // 用户角色集合
        Set<String> roles = SecurityUtils.getRoles();
        userInfoVO.setRoles(roles);

        // 用户权限集合
        if (CollUtil.isNotEmpty(roles)) {
            Set<String> perms = permissionService.getRolePermsFromCache(roles);
            userInfoVO.setPerms(perms);
        }
        return userInfoVO;
    }

    /**
     * 判断指定 username 是否属于当前用户（允许本人保留原用户名）
     */
    private boolean isCurrentUser(String username, Long userId) {
        SysUser currentUser = this.getOne(new LambdaQueryWrapper<SysUser>()
                .eq(SysUser::getId, userId)
                .select(SysUser::getUsername)
        );
        return currentUser != null && username.equals(currentUser.getUsername());
    }


}

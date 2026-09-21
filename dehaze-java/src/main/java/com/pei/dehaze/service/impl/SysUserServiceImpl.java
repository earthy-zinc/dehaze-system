package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.collection.CollectionUtil;
import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.annotation.AuditLog;
import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.DateUtils;
import com.pei.dehaze.common.util.IdUtils;
import com.pei.dehaze.converter.UserConverter;
import com.pei.dehaze.mapper.SysMemberMapper;
import com.pei.dehaze.mapper.SysUserMapper;
import com.pei.dehaze.model.read.UserRead;
import com.pei.dehaze.model.dto.UserAuthInfo;
import com.pei.dehaze.model.entity.SysMember;
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
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.data.redis.core.Cursor;
import org.springframework.data.redis.core.ScanOptions;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * 用户业务实现类
 *
 * @author earthyzinc
 * @since 2022/1/14
 */
@Service
@RequiredArgsConstructor
public class SysUserServiceImpl extends ServiceImpl<SysUserMapper, SysUser> implements SysUserService {

    /**
     * 超级管理员用户名（受删除/禁用保护，对齐 python user_service）
     */
    private static final String ROOT_USERNAME = "root";

    private final PasswordEncoder passwordEncoder;

    private final SysUserRoleService userRoleService;

    private final UserConverter userConverter;

    private final SysMenuService menuService;

    private final SysRoleService roleService;

    private final PermissionService permissionService;

    private final SysMemberMapper memberMapper;

    private final StringRedisTemplate redisTemplate;

    /**
     * 新用户默认密码（由各 profile 的 system.default-password 注入，源为根 .env 的 DEFAULT_PASSWORD）
     */
    @Value("${system.default-password}")
    private String defaultPassword;

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
        Page<UserPageVO> voPage = userConverter.read2PageVo(userPage);

        // 批量聚合会员信息（in userIds，避免 N+1）
        List<Long> userIds = voPage.getRecords().stream().map(UserPageVO::getId).toList();
        if (!userIds.isEmpty()) {
            Map<Long, SysMember> memberMap = memberMapper.selectList(new LambdaQueryWrapper<SysMember>()
                            .in(SysMember::getUserId, userIds))
                    .stream()
                    .collect(Collectors.toMap(SysMember::getUserId, Function.identity()));
            voPage.getRecords().forEach(vo -> fillMemberFields(vo, memberMap.get(vo.getId())));
        }
        return voPage;
    }

    /**
     * 填充分页 VO 的会员字段；quotaUsage 为 8 类任务本月 used/quota 求和（used/total）
     */
    private void fillMemberFields(UserPageVO vo, SysMember member) {
        if (member == null) {
            vo.setQuotaUsage("0/0");
            return;
        }
        vo.setMemberLevel(member.getLevelCode());
        vo.setMemberExpireTime(member.getExpireTime());
        int total = member.getMonthlyDehazeQuota() + member.getMonthlyDerainQuota() + member.getMonthlyDesnowQuota()
                + member.getMonthlyLowlightQuota() + member.getMonthlySuperResolutionQuota()
                + member.getMonthlyDenoiseQuota() + member.getMonthlyInpaintQuota() + member.getMonthlyEvaluateQuota();
        int used = member.getMonthlyDehazeUsed() + member.getMonthlyDerainUsed() + member.getMonthlyDesnowUsed()
                + member.getMonthlyLowlightUsed() + member.getMonthlySuperResolutionUsed()
                + member.getMonthlyDenoiseUsed() + member.getMonthlyInpaintUsed() + member.getMonthlyEvaluateUsed();
        vo.setQuotaUsage(used + "/" + total);
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

        // 业务白名单：用户名查全表判重（含软删行），删除后永久不可复用，理由见 SysUserMapper#countByUsernameAllDeleted
        long count = this.baseMapper.countByUsernameAllDeleted(username);
        if (count > 0) {
            throw new BusinessException(ResultCode.DATA_EXISTS, "该用户名不可用");
        }

        // 实体转换 form->entity
        SysUser entity = userConverter.form2Entity(userForm);

        // 设置默认加密密码（来自环境变量注入，不随代码硬编码）
        String defaultEncryptPwd = passwordEncoder.encode(defaultPassword);
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
     * 更新用户 — 用户名查重走白名单逻辑（查全表含软删行，理由见 SysUserMapper#countByUsernameAllDeleted）。
     *
     * @param userId   用户ID
     * @param userForm 用户表单对象
     * @return
     */
    @Override
    @Transactional(rollbackFor = Exception.class)
    @CacheEvict(value = "user:auth", allEntries = true)
    public boolean updateUser(Long userId, UserForm userForm) {

        SysUser existing = this.getById(userId);
        if (existing == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "用户不存在");
        }

        // 用户名字段只读，不可修改（与角色编码创建后不可修改保持一致）
        String username = userForm.getUsername();
        if (username != null && !username.equals(existing.getUsername())) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "用户名不可修改");
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
    @CacheEvict(value = "user:auth", allEntries = true)
    public boolean deleteUsers(String idsStr) {
        if (StrUtil.isBlank(idsStr)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "删除的用户数据为空");
        }
        List<Long> ids = IdUtils.parseIdList(idsStr);

        // 不可删除自己
        Long currentUserId = SecurityUtils.getUserId();
        if (currentUserId != null && ids.contains(currentUserId)) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "不可删除自己");
        }

        // 超级管理员受保护，不可删除
        List<SysUser> users = this.listByIds(ids);
        boolean hasRoot = users.stream().anyMatch(u -> ROOT_USERNAME.equals(u.getUsername()));
        if (hasRoot) {
            throw new BusinessException(ResultCode.ROOT_USER_PROTECTED, "超级管理员不可删除");
        }

        boolean result = this.removeByIds(ids);
        if (result) {
            // 删除后踢出目标用户全部在线会话（对齐 python user_service.delete_users）
            kickUserSessions(ids);
        }
        return result;
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
    @CacheEvict(value = "user:auth", allEntries = true)
    public boolean updatePassword(Long userId, String password) {
        validatePasswordComplexity(password);
        SysUser user = this.getById(userId);
        if (user == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "用户不存在");
        }
        Long currentUserId = SecurityUtils.getUserId();
        boolean result = this.update(new LambdaUpdateWrapper<SysUser>()
                .eq(SysUser::getId, userId)
                .set(SysUser::getPassword, passwordEncoder.encode(password))
                .set(SysUser::getUpdateBy, currentUserId)
        );
        if (result) {
            // 重置后踢出该用户全部在线会话，强制重新登录（对齐 python user_service.update_password）
            kickUserSessions(List.of(userId));
        }
        return result;
    }

    /**
     * 密码复杂度：8-20 位且必须同时包含字母与数字（对齐 python validate_password_complexity）
     */
    private void validatePasswordComplexity(String password) {
        if (password == null || password.length() < 8 || password.length() > 20
                || !password.matches(".*[a-zA-Z].*") || !password.matches(".*\\d.*")) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "密码必须包含字母和数字，8-20位");
        }
    }

    /**
     * 修改用户状态：用户不存在 A0401；禁用超级管理员 A0505（防自锁）；不可禁用自己 A0503
     */
    @Override
    @CacheEvict(value = "user:auth", allEntries = true)
    public boolean updateUserStatus(Long userId, Integer status) {
        SysUser user = this.getById(userId);
        if (user == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "用户不存在");
        }
        if (ROOT_USERNAME.equals(user.getUsername()) && status == 0) {
            throw new BusinessException(ResultCode.ROOT_USER_PROTECTED, "超级管理员不可禁用");
        }
        Long currentUserId = SecurityUtils.getUserId();
        if (currentUserId != null && currentUserId.equals(userId)) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "不可禁用自己");
        }
        boolean result = this.update(new LambdaUpdateWrapper<SysUser>()
                .eq(SysUser::getId, userId)
                .set(SysUser::getStatus, status)
                .set(SysUser::getUpdateBy, currentUserId)
        );
        if (result && status == 0) {
            // 禁用后实时踢出该用户全部在线会话，保证禁用立即生效（对齐 python user_service.update_user_status）
            kickUserSessions(List.of(userId));
        }
        return result;
    }

    /**
     * 踢出目标用户全部在线会话（对齐 python auth_service.kick_user_sessions_batch）。
     * 超级管理员会话不可被踢出（与 kickSession 语义一致）。
     */
    private void kickUserSessions(Collection<Long> userIds) {
        if (CollUtil.isEmpty(userIds)) {
            return;
        }
        List<String> sessionKeys = new ArrayList<>();
        Map<Long, List<String>> kickedByUser = new HashMap<>();
        try (Cursor<String> cursor = redisTemplate.scan(ScanOptions.scanOptions()
                .match(SecurityConstants.SESSION_PREFIX + "*").build())) {
            while (cursor.hasNext()) {
                String key = cursor.next();
                if (key.startsWith(SecurityConstants.SESSION_USER_PREFIX)) {
                    continue;
                }
                String raw = redisTemplate.opsForValue().get(key);
                if (raw == null) {
                    continue;
                }
                JSONObject data = JSONUtil.parseObj(raw);
                Long userId = data.getLong("userId");
                if (userId == null || !userIds.contains(userId)) {
                    continue;
                }
                JSONArray authorities = data.getJSONArray("authorities");
                if (authorities != null && authorities.contains(SecurityConstants.ROLE_PREFIX + "ROOT")) {
                    continue;
                }
                sessionKeys.add(key);
                kickedByUser.computeIfAbsent(userId, id -> new ArrayList<>())
                        .add(key.substring(SecurityConstants.SESSION_PREFIX.length()));
            }
        }
        if (sessionKeys.isEmpty()) {
            return;
        }
        redisTemplate.delete(sessionKeys);
        // 同步清理多点登录索引（session:user:{userId} ZSet 中移除被踢会话元素）
        for (Map.Entry<Long, List<String>> entry : kickedByUser.entrySet()) {
            redisTemplate.opsForZSet().remove(
                    SecurityConstants.SESSION_USER_PREFIX + entry.getKey(), entry.getValue().toArray());
        }
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

}

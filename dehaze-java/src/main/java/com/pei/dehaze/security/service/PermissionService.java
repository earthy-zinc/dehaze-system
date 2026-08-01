package com.pei.dehaze.security.service;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.SysRoleMenuService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;
import org.springframework.util.PatternMatchUtils;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.stream.Collectors;

/**
 * SpringSecurity 权限校验
 *
 * @author earthyzinc
 * @since 2022/2/22
 */
@Component("ss")
@RequiredArgsConstructor
@Slf4j
public class PermissionService {

    private final StringRedisTemplate stringRedisTemplate;

    private final SysRoleMenuService sysRoleMenuService;

    /**
     * SingleFlight 等待超时时间（秒）
     */
    private static final long SINGLEFLIGHT_TIMEOUT_SECONDS = 30;

    /**
     * SingleFlight 最大重试次数
     */
    private static final int SINGLEFLIGHT_MAX_RETRIES = 3;

    /**
     * SingleFlight 飞行表：合并同一 roleCode 的并发回源请求，防止缓存击穿
     */
    private final ConcurrentHashMap<String, CompletableFuture<Set<String>>> singleFlightMap = new ConcurrentHashMap<>();

    /**
     * 判断当前登录用户是否拥有操作权限
     *
     * @param requiredPerm 所需权限
     * @return 是否有权限
     */
    public boolean hasPerm(String requiredPerm) {

        if (CharSequenceUtil.isBlank(requiredPerm)) {
            return false;
        }
        // 超级管理员放行
        if (SecurityUtils.isRoot()) {
            return true;
        }

        // 获取当前登录用户的角色编码集合
        Set<String> roleCodes = SecurityUtils.getRoles();
        if (CollUtil.isEmpty(roleCodes)) {
            return false;
        }

        // 获取当前登录用户的所有角色的权限列表
        Set<String> rolePerms = this.getRolePermsFromCache(roleCodes);
        if (CollUtil.isEmpty(rolePerms)) {
            return false;
        }
        // 判断当前登录用户的所有角色的权限列表中是否包含所需权限
        boolean hasPermission = rolePerms.stream()
                .anyMatch(rolePerm ->
                        // 匹配权限，支持通配符(* 等)
                        PatternMatchUtils.simpleMatch(rolePerm, requiredPerm)
                );

        if (!hasPermission) {
            log.debug("用户无操作权限");
        }
        return hasPermission;
    }


    /**
     * 从缓存中获取角色权限列表，缓存未命中时回源数据库并回填缓存。
     * <p>
     * 逐角色独立 Key + 独立 TTL，避免单角色刷新时重置全部角色的 TTL。
     * 使用 SingleFlight 防止缓存击穿（同一 roleCode 的并发回源请求合并为一个）。
     *
     * @param roleCodes 角色编码集合
     * @return 角色权限列表
     */
    public Set<String> getRolePermsFromCache(Set<String> roleCodes) {
        if (CollUtil.isEmpty(roleCodes)) {
            return Collections.emptySet();
        }

        List<String> roleCodeList = new ArrayList<>(roleCodes);
        List<String> keys = roleCodeList.stream()
                .map(code -> SecurityConstants.ROLE_PERMS_PREFIX + code)
                .collect(Collectors.toList());

        Set<String> perms = new HashSet<>();
        List<String> missingRoleCodes = new ArrayList<>();

        // 1. 批量查询逐角色独立 Key
        List<String> rolePermsList = stringRedisTemplate.opsForValue().multiGet(keys);
        if (CollUtil.isEmpty(rolePermsList)) {
            rolePermsList = Collections.emptyList();
        }

        for (int i = 0; i < roleCodeList.size(); i++) {
            String rolePermsJson = rolePermsList.get(i);
            if (rolePermsJson != null) {
                Set<String> rolePerms = JSONUtil.parseArray(rolePermsJson)
                        .stream()
                        .map(Object::toString)
                        .collect(Collectors.toSet());
                perms.addAll(rolePerms);
            } else {
                // 该角色权限缺失，需要回源
                missingRoleCodes.add(roleCodeList.get(i));
            }
        }

        // 2. 对缺失的角色逐个回源并回填（SingleFlight 防击穿）
        for (String roleCode : missingRoleCodes) {
            Set<String> rolePerms = loadRolePermsWithSingleFlight(roleCode);
            perms.addAll(rolePerms);
        }

        return perms;
    }

    /**
     * SingleFlight 加载角色权限：同一 roleCode 的并发请求合并为一个回源操作。
     * 回源后通过 {@link SysRoleMenuService#refreshRolePermsCache(String)} 回填缓存并设置独立 TTL。
     *
     * @param roleCode 角色编码
     * @return 该角色的权限集合（回源失败时返回空集合，不影响其他角色的鉴权）
     */
    private Set<String> loadRolePermsWithSingleFlight(String roleCode) {
        int retries = 0;
        while (true) {
            // 检查是否有正在进行的加载
            CompletableFuture<Set<String>> existing = singleFlightMap.get(roleCode);
            if (existing != null) {
                try {
                    return existing.get(SINGLEFLIGHT_TIMEOUT_SECONDS, TimeUnit.SECONDS);
                } catch (TimeoutException e) {
                    log.warn("SingleFlight 等待超时: roleCode={}", roleCode, e);
                    return Collections.emptySet();
                } catch (Exception e) {
                    // 前一个加载失败，移除并重试
                    singleFlightMap.remove(roleCode, existing);
                    if (++retries > SINGLEFLIGHT_MAX_RETRIES) {
                        throw new BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR);
                    }
                    continue;
                }
            }

            // 尝试创建新的加载任务
            CompletableFuture<Set<String>> newFuture = new CompletableFuture<>();
            CompletableFuture<Set<String>> raced = singleFlightMap.putIfAbsent(roleCode, newFuture);
            if (raced != null) {
                // 另一个线程先创建了 future，等待它
                try {
                    return raced.get(SINGLEFLIGHT_TIMEOUT_SECONDS, TimeUnit.SECONDS);
                } catch (TimeoutException e) {
                    log.warn("SingleFlight 等待超时: roleCode={}", roleCode, e);
                    return Collections.emptySet();
                } catch (Exception e) {
                    singleFlightMap.remove(roleCode, raced);
                    if (++retries > SINGLEFLIGHT_MAX_RETRIES) {
                        throw new BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR);
                    }
                    continue;
                }
            }

            // 当前线程负责回源并回填
            try {
                // refreshRolePermsCache 会先 delete 再 set，并设置独立 TTL
                sysRoleMenuService.refreshRolePermsCache(roleCode);
                // 从 Redis 读取回填的权限（纯 JSON 字符串数组）
                String rolePermsJson = stringRedisTemplate.opsForValue()
                        .get(SecurityConstants.ROLE_PERMS_PREFIX + roleCode);
                Set<String> rolePerms = new HashSet<>();
                if (rolePermsJson != null) {
                    rolePerms = JSONUtil.parseArray(rolePermsJson)
                            .stream()
                            .map(Object::toString)
                            .collect(Collectors.toSet());
                }
                newFuture.complete(rolePerms);
                return rolePerms;
            } catch (Exception e) {
                newFuture.completeExceptionally(e);
                log.error("加载角色权限失败: roleCode={}", roleCode, e);
                throw new BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR.getMsg(), e);
            } finally {
                singleFlightMap.remove(roleCode, newFuture);
            }
        }
    }

}

package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.mapper.SysRoleMenuMapper;
import com.pei.dehaze.model.bo.RolePermsBO;
import com.pei.dehaze.model.entity.SysRoleMenu;
import com.pei.dehaze.service.SysRoleMenuService;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.List;
import java.util.Set;


/**
 * 角色菜单业务实现
 *
 * @author earthyzinc
 * @since 2.5.0
 */
@Service
@RequiredArgsConstructor
public class SysRoleMenuServiceImpl extends ServiceImpl<SysRoleMenuMapper, SysRoleMenu> implements SysRoleMenuService {

    private final RedisTemplate<String, Object> redisTemplate;

    /**
     * 权限缓存 TTL（30 分钟）
     */
    private static final Duration ROLE_PERMS_TTL = Duration.ofMinutes(30);

    /**
     * 初始化权限缓存
     */
    @PostConstruct
    public void initRolePermsCache() {
        refreshRolePermsCache();
    }

    /**
     * 构造逐角色独立 Redis Key（独立 TTL，避免单角色刷新重置全部角色 TTL）
     */
    private String rolePermsKey(String roleCode) {
        return SecurityConstants.ROLE_PERMS_PREFIX + roleCode;
    }

    /**
     * 刷新权限缓存（全量）
     * 逐角色独立 Key + 独立 TTL，避免单角色刷新时重置全部角色的 TTL。
     */
    @Override
    public void refreshRolePermsCache() {
        // 清理所有角色权限缓存 Key
        Set<String> keys = redisTemplate.keys(SecurityConstants.ROLE_PERMS_PREFIX + "*");
        if (CollUtil.isNotEmpty(keys)) {
            redisTemplate.delete(keys);
        }

        List<RolePermsBO> list = this.baseMapper.getRolePermsList(null);
        if (CollUtil.isNotEmpty(list)) {
            list.forEach(item -> {
                String roleCode = item.getRoleCode();
                Set<String> perms = item.getPerms();
                redisTemplate.opsForValue().set(rolePermsKey(roleCode), perms, ROLE_PERMS_TTL);
            });
        }
    }

    /**
     * 刷新权限缓存（指定角色）
     */
    @Override
    public void refreshRolePermsCache(String roleCode) {
        redisTemplate.delete(rolePermsKey(roleCode));

        List<RolePermsBO> list = this.baseMapper.getRolePermsList(roleCode);
        if (CollUtil.isNotEmpty(list)) {
            RolePermsBO rolePerms = list.get(0);
            if (rolePerms == null) {
                return;
            }
            Set<String> perms = rolePerms.getPerms();
            redisTemplate.opsForValue().set(rolePermsKey(roleCode), perms, ROLE_PERMS_TTL);
        }
    }

    /**
     * 刷新权限缓存 (角色编码变更时调用)
     * 删除旧编码缓存后，按新编码刷新（复用单参数逻辑避免重复 DB 查询 + Redis 写入）
     */
    @Override
    public void refreshRolePermsCache(String oldRoleCode, String newRoleCode) {
        redisTemplate.delete(rolePermsKey(oldRoleCode));
        refreshRolePermsCache(newRoleCode);
    }


    /**
     * 获取角色拥有的菜单ID集合
     *
     * @param roleId 角色ID
     * @return 菜单ID集合
     */
    @Override
    public List<Long> listMenuIdsByRoleId(Long roleId) {
        return this.baseMapper.listMenuIdsByRoleId(roleId);
    }

}

package com.pei.dehaze.security.service;

import com.pei.dehaze.model.dto.UserAuthInfo;
import com.pei.dehaze.security.model.SysUserDetails;
import com.pei.dehaze.service.SysUserService;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

/**
 * 系统用户认证
 *
 * @author earthyzinc
 */
@Service
@RequiredArgsConstructor
public class SysUserDetailsService implements UserDetailsService {

    private final SysUserService sysUserService;

    /**
     * 加载用户认证信息（带缓存，TTL 由 CacheManager 统一管理）
     * <p>
     * 缓存 key 为 username，避免每次登录都查库。
     * 用户信息变更时通过 SysUserService 中的 @CacheEvict 清除。
     */
    @Override
    @Cacheable(value = "user:auth", key = "#username")
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {

        UserAuthInfo userAuthInfo = sysUserService.getUserAuthInfo(username);
        if (userAuthInfo == null) {
            throw new UsernameNotFoundException("用户不存在: " + username);
        }
        return new SysUserDetails(userAuthInfo);
    }
}

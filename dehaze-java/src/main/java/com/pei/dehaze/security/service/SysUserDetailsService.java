package com.pei.dehaze.security.service;

import com.pei.dehaze.model.dto.UserAuthInfo;
import com.pei.dehaze.security.model.SysUserDetails;
import com.pei.dehaze.service.SysUserService;
import lombok.RequiredArgsConstructor;
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
     * 加载用户认证信息。
     * <p>
     * 缓存由 {@link SysUserService#getUserAuthInfo(String)} 上的 @Cacheable 提供，
     * 缓存对象为 {@link UserAuthInfo}（纯 POJO，仅含 Long/String/Integer/Set&lt;String&gt; 字段，Jackson 友好）。
     * <p>
     * 不直接缓存 {@link SysUserDetails} 的原因：其 authorities 字段为 {@code Set<SimpleGrantedAuthority>}，
     * SimpleGrantedAuthority 是 Spring Security 的 final 类，无默认构造器、无 @JsonCreator，
     * Jackson 无法反序列化；缓存 SysUserDetails 会导致读取时反序列化失败被误判为"密码错误"，
     * 连锁触发账号锁定。SysUserDetails 改为在缓存命中后于内存中由 UserAuthInfo 构造。
     */
    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        UserAuthInfo userAuthInfo = sysUserService.getUserAuthInfo(username);
        if (userAuthInfo == null) {
            throw new UsernameNotFoundException("用户不存在: " + username);
        }
        return new SysUserDetails(userAuthInfo);
    }
}

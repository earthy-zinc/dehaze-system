package com.pei.dehaze.security.util;

import cn.hutool.core.convert.Convert;
import cn.hutool.core.util.IdUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.jwt.JWTUtil;
import cn.hutool.jwt.RegisteredPayload;
import com.pei.dehaze.common.constant.JwtClaimConstants;
import com.pei.dehaze.config.property.SecurityProperties;
import com.pei.dehaze.security.model.SysUserDetails;
import lombok.RequiredArgsConstructor;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.stereotype.Component;

import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * JWT Token 工具类
 *
 * @author Ray Hao
 * @since 2.6.0
 */
@Component
@RequiredArgsConstructor
public class JwtUtils {

    private final SecurityProperties securityProperties;

    /**
     * 生成 JWT Token
     *
     * @param authentication 用户认证信息
     * @return Token 字符串
     */
    public String createToken(Authentication authentication) {

        SysUserDetails userDetails = (SysUserDetails) authentication.getPrincipal();

        Map<String, Object> payload = new HashMap<>();
        payload.put(JwtClaimConstants.USER_ID, userDetails.getUserId()); // 用户ID
        payload.put(JwtClaimConstants.DEPT_ID, userDetails.getDeptId()); // 部门ID
        payload.put(JwtClaimConstants.DATA_SCOPE, userDetails.getDataScope()); // 数据权限范围

        // claims 中添加角色信息
        Set<String> roles = userDetails.getAuthorities().stream()
                .map(GrantedAuthority::getAuthority)
                .collect(Collectors.toSet());
        payload.put(JwtClaimConstants.AUTHORITIES, roles);


        Date now = new Date();
        long ttlSeconds = securityProperties.getJwt().getTtl();
        Date expiration = new Date(now.getTime() + ttlSeconds * 1000L);
        payload.put(RegisteredPayload.ISSUED_AT, now);
        payload.put(RegisteredPayload.EXPIRES_AT, expiration);
        payload.put(RegisteredPayload.SUBJECT, authentication.getName());
        payload.put(RegisteredPayload.JWT_ID, IdUtil.simpleUUID());

        return JWTUtil.createToken(payload, securityProperties.getJwt().getKey().getBytes());
    }


    /**
     * 从 JWT Token 中解析 Authentication  用户认证信息
     *
     * @param payloads JWT 载体
     * @return 用户认证信息
     */
    public static UsernamePasswordAuthenticationToken getAuthentication(JSONObject payloads) {
        SysUserDetails userDetails = new SysUserDetails();
        userDetails.setUserId(payloads.getLong(JwtClaimConstants.USER_ID)); // 用户ID
        userDetails.setDeptId(payloads.getLong(JwtClaimConstants.DEPT_ID)); // 部门ID
        userDetails.setDataScope(payloads.getInt(JwtClaimConstants.DATA_SCOPE)); // 数据权限范围

        userDetails.setUsername(payloads.getStr(RegisteredPayload.SUBJECT)); // 用户名
        // 角色集合
        Set<SimpleGrantedAuthority> authorities = new HashSet<>();
        JSONArray jsonArray = payloads.getJSONArray(JwtClaimConstants.AUTHORITIES);
        if (jsonArray != null) {
            authorities = jsonArray.stream()
                    .map(authority -> new SimpleGrantedAuthority(Convert.toStr(authority)))
                    .collect(Collectors.toSet());
        }
        userDetails.setAuthorities(authorities);

        return new UsernamePasswordAuthenticationToken(userDetails, "", authorities);
    }


}

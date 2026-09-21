package com.pei.dehaze.service.impl;

import cn.hutool.captcha.generator.CodeGenerator;
import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.mapper.SysUserMapper;
import com.pei.dehaze.plugin.captcha.CaptchaProperties;
import com.pei.dehaze.service.LoginLogService;
import com.pei.dehaze.service.MemberService;
import com.pei.dehaze.service.SysRoleService;
import com.pei.dehaze.service.SysUserRoleService;
import com.pei.dehaze.service.SysUserService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ZSetOperations;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.awt.Font;
import java.util.LinkedHashSet;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * 多点登录设备数上限（F-AM-011）单元测试。
 * <p>
 * 索引 session:user:{userId} 为 ZSet（member=sessionId，score=登录 epoch 秒）；断言落在可观测行为上：
 * 超限时最早会话的 Redis 键被删除、索引元素被移除、上限按"管理员固定 10 台 / 普通用户取等级权益"取值。
 */
@DisplayName("AuthServiceImpl 同时在线设备数上限单元测试")
@ExtendWith(MockitoExtension.class)
class AuthServiceImplSessionLimitTest {

    private static final Long USER_ID = 9L;
    private static final String INDEX_KEY = SecurityConstants.SESSION_USER_PREFIX + USER_ID;

    @Mock
    private AuthenticationManager authenticationManager;
    @Mock
    private StringRedisTemplate redisTemplate;
    @Mock
    private CodeGenerator codeGenerator;
    @Mock
    private CaptchaProperties captchaProperties;
    @Mock
    private PasswordEncoder passwordEncoder;
    @Mock
    private SysUserService sysUserService;
    @Mock
    private SysUserMapper sysUserMapper;
    @Mock
    private SysRoleService sysRoleService;
    @Mock
    private SysUserRoleService sysUserRoleService;
    @Mock
    private LoginLogService loginLogService;
    @Mock
    private MemberService memberService;
    @Mock
    private ZSetOperations<String, String> zset;

    private AuthServiceImpl service;

    @BeforeEach
    void setUp() {
        service = new AuthServiceImpl(authenticationManager, redisTemplate, codeGenerator,
                new Font("Arial", Font.PLAIN, 12), captchaProperties, passwordEncoder,
                sysUserService, sysUserMapper, sysRoleService, sysUserRoleService,
                loginLogService, memberService);
    }

    private List<SimpleGrantedAuthority> authorities(String... roles) {
        return List.of(roles).stream().map(SimpleGrantedAuthority::new).toList();
    }

    @Test
    @DisplayName("普通用户超限：踢掉最早登录会话，新会话保留在索引中")
    void normalUserOverLimitEvictsEarliest() {
        when(redisTemplate.opsForZSet()).thenReturn(zset);
        when(memberService.getMaxDevices(USER_ID)).thenReturn(2);
        when(zset.zCard(INDEX_KEY)).thenReturn(3L);
        when(zset.range(INDEX_KEY, 0, -1))
                .thenReturn(new LinkedHashSet<>(List.of("s-oldest", "s-middle", "s-newest")));

        service.enforceDeviceLimit("s-brand-new", USER_ID, authorities("ROLE_GUEST"));

        // 最早会话的会话键被删除（其下一次请求 401），索引元素同步移除
        verify(redisTemplate).delete(List.of(SecurityConstants.SESSION_PREFIX + "s-oldest"));
        ArgumentCaptor<Object[]> removedCaptor = ArgumentCaptor.forClass(Object[].class);
        verify(zset).remove(eq(INDEX_KEY), removedCaptor.capture());
        assertThat(removedCaptor.getValue()).containsExactly("s-oldest");

        // 新会话以整秒 epoch 作为 score 入索引（DATETIME 秒精度口径）
        ArgumentCaptor<Double> scoreCaptor = ArgumentCaptor.forClass(Double.class);
        verify(zset).add(eq(INDEX_KEY), eq("s-brand-new"), scoreCaptor.capture());
        double score = scoreCaptor.getValue();
        assertThat(score).isEqualTo(Math.floor(score));
    }

    @Test
    @DisplayName("普通用户未超限：不删除任何会话键")
    void normalUserWithinLimitKeepsAll() {
        when(redisTemplate.opsForZSet()).thenReturn(zset);
        when(memberService.getMaxDevices(USER_ID)).thenReturn(3);
        when(zset.zCard(INDEX_KEY)).thenReturn(3L);

        service.enforceDeviceLimit("s-new", USER_ID, authorities("ROLE_GUEST"));

        verify(redisTemplate, never()).delete(anyList());
        verify(zset, never()).remove(any(), any());
    }

    @Test
    @DisplayName("管理员固定 10 台：不查会员等级权益，11 台在线只踢最早 1 台")
    void adminFixedTenSkipsMemberLookup() {
        when(redisTemplate.opsForZSet()).thenReturn(zset);
        when(zset.zCard(INDEX_KEY)).thenReturn(11L);
        LinkedHashSet<String> members = new LinkedHashSet<>();
        for (int i = 0; i < 11; i++) {
            members.add("s-admin-" + i);
        }
        when(zset.range(INDEX_KEY, 0, -1)).thenReturn(members);

        service.enforceDeviceLimit("s-admin-new", USER_ID, authorities("ROLE_ADMIN"));

        verifyNoInteractions(memberService);
        verify(redisTemplate).delete(List.of(SecurityConstants.SESSION_PREFIX + "s-admin-0"));
    }

    @Test
    @DisplayName("会话索引继承 7 天 TTL（与 session:{id} 同口径）")
    void indexKeyGetsSessionTtl() {
        when(redisTemplate.opsForZSet()).thenReturn(zset);
        when(memberService.getMaxDevices(USER_ID)).thenReturn(1);
        when(zset.zCard(INDEX_KEY)).thenReturn(1L);

        service.enforceDeviceLimit("s-only", USER_ID, authorities("ROLE_GUEST"));

        verify(redisTemplate).expire(INDEX_KEY, SecurityConstants.SESSION_TTL, java.util.concurrent.TimeUnit.SECONDS);
        assertThat(SecurityConstants.ADMIN_MAX_DEVICES).isEqualTo(10);
    }
}

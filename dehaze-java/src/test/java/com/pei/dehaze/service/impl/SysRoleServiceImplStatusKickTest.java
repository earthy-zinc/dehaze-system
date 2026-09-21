package com.pei.dehaze.service.impl;

import com.pei.dehaze.common.constant.SecurityConstants;
import com.pei.dehaze.converter.RoleConverter;
import com.pei.dehaze.mapper.SysMenuMapper;
import com.pei.dehaze.model.entity.SysRole;
import com.pei.dehaze.service.SysRoleMenuService;
import com.pei.dehaze.service.SysUserRoleService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * 角色状态变更权限传播测试：禁用角色踢出关联在线用户，启用不踢
 */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
@DisplayName("角色状态变更踢出在线用户测试")
class SysRoleServiceImplStatusKickTest {

    @Mock
    private SysRoleMenuService roleMenuService;
    @Mock
    private SysUserRoleService userRoleService;
    @Mock
    private SysMenuMapper menuMapper;
    @Mock
    private RoleConverter roleConverter;
    @Mock
    private StringRedisTemplate stringRedisTemplate;
    @Mock
    private ValueOperations<String, String> valueOperations;

    @InjectMocks
    private SysRoleServiceImpl service;

    private SysRoleServiceImpl spyService(SysRole role) {
        SysRoleServiceImpl spy = org.mockito.Mockito.spy(service);
        doReturn(role).when(spy).getById(role.getId());
        doReturn(true).when(spy).updateById(any(SysRole.class));
        return spy;
    }

    private SysRole stubRole() {
        SysRole role = new SysRole();
        role.setId(1L);
        role.setCode("TEST");
        role.setStatus(1);
        return role;
    }

    @Test
    @DisplayName("禁用角色：踢出关联在线会话")
    void disableRoleKicksOnlineSessions() {
        SysRole role = stubRole();
        SysRoleServiceImpl spy = spyService(role);

        when(userRoleService.listUsernamesByRoleIds(List.of(1L))).thenReturn(List.of("zhangsan"));
        when(stringRedisTemplate.keys(SecurityConstants.SESSION_PREFIX + "*")).thenReturn(Set.of("session:abc"));
        when(stringRedisTemplate.opsForValue()).thenReturn(valueOperations);
        when(valueOperations.get("session:abc"))
                .thenReturn("{\"userId\":9,\"username\":\"zhangsan\",\"authorities\":[]}");

        assertTrue(spy.updateRoleStatus(1L, 0));

        verify(stringRedisTemplate).delete("session:abc");
        verify(roleMenuService).refreshRolePermsCache("TEST");
    }

    @Test
    @DisplayName("启用角色：不踢会话")
    void enableRoleDoesNotKickSessions() {
        SysRole role = stubRole();
        SysRoleServiceImpl spy = spyService(role);

        assertTrue(spy.updateRoleStatus(1L, 1));

        verify(stringRedisTemplate, never()).keys(anyString());
        verify(stringRedisTemplate, never()).delete("session:abc");
    }

    @Test
    @DisplayName("禁用角色：超级管理员会话不被踢出")
    void disableRoleSparesRootSessions() {
        SysRole role = stubRole();
        SysRoleServiceImpl spy = spyService(role);

        when(userRoleService.listUsernamesByRoleIds(List.of(1L))).thenReturn(List.of("root"));
        when(stringRedisTemplate.keys(SecurityConstants.SESSION_PREFIX + "*")).thenReturn(Set.of("session:root"));
        when(stringRedisTemplate.opsForValue()).thenReturn(valueOperations);
        when(valueOperations.get("session:root"))
                .thenReturn("{\"userId\":1,\"username\":\"root\",\"authorities\":[\"ROLE_ROOT\"]}");

        assertTrue(spy.updateRoleStatus(1L, 0));

        verify(stringRedisTemplate, never()).delete("session:root");
    }
}

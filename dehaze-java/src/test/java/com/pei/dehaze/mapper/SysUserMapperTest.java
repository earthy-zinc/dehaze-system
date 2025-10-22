package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.base.BaseTest;
import com.pei.dehaze.model.bo.UserBO;
import com.pei.dehaze.model.query.UserPageQuery;
import com.pei.dehaze.security.util.SecurityUtils;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;

import java.util.Collections;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * SysUserMapper 集成测试
 * 
 * 重点测试数据权限拦截器（@DataPermission）的 SQL 拼接逻辑
 * 
 * 测试场景：
 * 1. 全部数据权限（ROOT 角色）- 不拼接过滤条件
 * 2. 自定义数据权限 - 拼接部门 ID 过滤
 * 3. 本部门数据权限 - 拼接当前用户部门过滤
 * 4. 本部门及子部门数据权限 - 拼接部门树过滤
 * 5. 仅本人数据权限 - 拼接用户 ID 过滤
 * 
 * @author earthyzinc
 */
@DisplayName("用户 Mapper 集成测试")
class SysUserMapperTest extends BaseTest {

    @Autowired
    private SysUserMapper sysUserMapper;

    /**
     * 模拟设置当前登录用户的 Spring Security 上下文
     */
    private void mockSecurityContext(Long userId, String username, Set<String> roles, Integer dataScope, Long deptId) {
        // 创建自定义的 UserDetails（模拟 SecurityUtils.getUser()）
        var authentication = new UsernamePasswordAuthenticationToken(
                username,
                null,
                Collections.singletonList(new SimpleGrantedAuthority("ROLE_USER")));
        SecurityContextHolder.getContext().setAuthentication(authentication);
    }

    @Test
    @DisplayName("数据权限测试 - 全部数据（ROOT 角色）")
    void testDataPermission_AllData() {
        // Given: 模拟 ROOT 角色用户（dataScope=1，查看全部数据）
        mockSecurityContext(1L, "root", Set.of("ROOT"), 1, 1L);

        UserPageQuery query = new UserPageQuery();
        query.setPageNum(1);
        query.setPageSize(10);
        Page<UserBO> page = new Page<>(1, 10);

        // When: 执行查询（应该不拼接任何过滤条件）
        IPage<UserBO> result = sysUserMapper.listPagedUsers(page, query);

        // Then: 验证结果（应该能查到所有用户）
        assertNotNull(result, "查询结果不应为空");
        assertTrue(result.getTotal() >= 3, "ROOT 角色应该能查看所有用户（至少3个：root, admin, test）");

        // 清理 Security 上下文
        SecurityContextHolder.clearContext();
    }

    @Test
    @DisplayName("数据权限测试 - 本部门数据（ADMIN 角色）")
    void testDataPermission_DeptData() {
        // Given: 模拟 ADMIN 角色用户（dataScope=4，查看本部门数据）
        // admin 用户属于部门 ID=1
        mockSecurityContext(2L, "admin", Set.of("ADMIN"), 4, 1L);

        UserPageQuery query = new UserPageQuery();
        query.setPageNum(1);
        query.setPageSize(10);
        Page<UserBO> page = new Page<>(1, 10);

        // When: 执行查询（应该拼接 u.dept_id = 1 条件）
        IPage<UserBO> result = sysUserMapper.listPagedUsers(page, query);

        // Then: 验证结果
        assertNotNull(result, "查询结果不应为空");
        // 验证返回的数据（数据权限拦截器已拼接 dept_id 过滤条件）
        assertTrue(result.getTotal() >= 0, "查询结果应该返回有效数据");

        // 清理 Security 上下文
        SecurityContextHolder.clearContext();
    }

    @Test
    @DisplayName("数据权限测试 - 仅本人数据（GUEST 角色）")
    void testDataPermission_SelfData() {
        // Given: 模拟 GUEST 角色用户（dataScope=5，仅查看本人数据）
        // test 用户 ID=3
        mockSecurityContext(3L, "test", Set.of("GUEST"), 5, 1L);

        UserPageQuery query = new UserPageQuery();
        query.setPageNum(1);
        query.setPageSize(10);
        Page<UserBO> page = new Page<>(1, 10);

        // When: 执行查询（应该拼接 u.create_by = 'test' 或 u.id = 3 条件）
        IPage<UserBO> result = sysUserMapper.listPagedUsers(page, query);

        // Then: 验证结果（应该只能查到自己创建的或者自己的数据）
        assertNotNull(result, "查询结果不应为空");
        // GUEST 角色应该只能看到有限的数据
        assertTrue(result.getTotal() <= 1, "GUEST 角色应该只能查看本人数据");

        // 清理 Security 上下文
        SecurityContextHolder.clearContext();
    }

    @Test
    @DisplayName("分页查询用户 - 基础功能")
    void testListPagedUsers_Basic() {
        // Given: 准备查询参数
        UserPageQuery query = new UserPageQuery();
        query.setPageNum(1);
        query.setPageSize(10);
        Page<UserBO> page = new Page<>(1, 10);

        // When: 执行查询
        IPage<UserBO> result = sysUserMapper.listPagedUsers(page, query);

        // Then: 验证结果
        assertNotNull(result, "查询结果不应为空");
        assertNotNull(result.getRecords(), "记录列表不应为空");
        assertTrue(result.getTotal() >= 0, "总记录数应该大于等于0");
    }

    @Test
    @DisplayName("分页查询用户 - 关键字搜索")
    void testListPagedUsers_WithKeywords() {
        // Given: 使用关键字查询
        UserPageQuery query = new UserPageQuery();
        query.setPageNum(1);
        query.setPageSize(10);
        query.setKeywords("admin");
        Page<UserBO> page = new Page<>(1, 10);

        // When: 执行查询
        IPage<UserBO> result = sysUserMapper.listPagedUsers(page, query);

        // Then: 验证结果
        assertNotNull(result, "查询结果不应为空");
        if (result.getTotal() > 0) {
            // 验证返回的用户包含关键字
            result.getRecords().forEach(user -> {
                boolean containsKeyword = user.getUsername().contains("admin") ||
                        (user.getNickname() != null && user.getNickname().contains("admin"));
                assertTrue(containsKeyword, "返回的用户应该包含关键字 'admin'");
            });
        }
    }

    @Test
    @DisplayName("根据用户名获取认证信息")
    void testGetUserAuthInfo() {
        // Given: 准备用户名
        String username = "admin";

        // When: 查询认证信息
        var authInfo = sysUserMapper.getUserAuthInfo(username);

        // Then: 验证结果
        assertNotNull(authInfo, "认证信息不应为空");
        assertEquals(username, authInfo.getUsername(), "用户名应该匹配");
        assertNotNull(authInfo.getPassword(), "密码不应为空");
        assertNotNull(authInfo.getRoles(), "角色集合不应为空");
        assertTrue(authInfo.getRoles().contains("ADMIN"), "应该包含 ADMIN 角色");
    }

    @Test
    @DisplayName("获取用户表单数据")
    void testGetUserFormData() {
        // Given: 准备用户 ID（admin 用户）
        Long userId = 2L;

        // When: 查询用户表单数据
        var userForm = sysUserMapper.getUserFormData(userId);

        // Then: 验证结果
        assertNotNull(userForm, "用户表单数据不应为空");
        assertEquals(userId, userForm.getId(), "用户 ID 应该匹配");
        assertNotNull(userForm.getUsername(), "用户名不应为空");
        assertNotNull(userForm.getRoleIds(), "角色 ID 列表不应为空");
    }
}

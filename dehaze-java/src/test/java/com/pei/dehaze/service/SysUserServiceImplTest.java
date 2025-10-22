package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.pei.dehaze.base.BaseTest;
import com.pei.dehaze.model.form.UserForm;
import com.pei.dehaze.model.query.UserPageQuery;
import com.pei.dehaze.model.vo.UserPageVO;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * SysUserService 单元测试
 * 
 * 测试场景：
 * 1. 用户分页查询
 * 2. 用户新增
 * 3. 用户更新
 * 4. 用户删除
 * 5. 密码修改
 * 6. 用户名重复校验
 * 
 * @author earthyzinc
 */
@DisplayName("用户服务测试")
class SysUserServiceImplTest extends BaseTest {

    @Autowired
    private SysUserService sysUserService;

    @Test
    @DisplayName("分页查询用户列表")
    void testListPagedUsers() {
        // Given: 准备查询参数
        UserPageQuery query = new UserPageQuery();
        query.setPageNum(1);
        query.setPageSize(10);
        query.setKeywords("test");

        // When: 执行分页查询
        IPage<UserPageVO> page = sysUserService.listPagedUsers(query);

        // Then: 验证结果
        assertNotNull(page, "分页结果不应为空");
        assertTrue(page.getTotal() >= 0, "总记录数应该大于等于0");
        assertNotNull(page.getRecords(), "记录列表不应为空");
    }

    @Test
    @DisplayName("新增用户 - 成功")
    void testSaveUser_Success() {
        // Given: 准备用户表单数据
        UserForm userForm = new UserForm();
        userForm.setUsername("testuser_" + System.currentTimeMillis());
        userForm.setNickname("测试用户");
        userForm.setMobile("13800138000");
        userForm.setGender(1);
        userForm.setEmail("test@example.com");
        userForm.setStatus(1);
        userForm.setDeptId(1L);
        userForm.setRoleIds(Arrays.asList(2L)); // ADMIN 角色

        // When: 执行新增操作
        boolean result = sysUserService.saveUser(userForm);

        // Then: 验证结果
        assertTrue(result, "新增用户应该成功");
    }

    @Test
    @DisplayName("新增用户 - 用户名重复")
    void testSaveUser_DuplicateUsername() {
        // Given: 使用已存在的用户名 (root)
        UserForm userForm = new UserForm();
        userForm.setUsername("root");
        userForm.setNickname("测试用户");
        userForm.setMobile("13800138000");
        userForm.setGender(1);
        userForm.setStatus(1);
        userForm.setDeptId(1L);
        userForm.setRoleIds(Arrays.asList(2L));

        // When & Then: 应该抛出异常
        assertThrows(IllegalArgumentException.class,
                () -> sysUserService.saveUser(userForm),
                "用户名重复应该抛出异常");
    }

    @Test
    @DisplayName("更新用户 - 成功")
    void testUpdateUser_Success() {
        // Given: 准备更新数据 (更新 test 用户，ID=3)
        Long userId = 3L;
        UserForm userForm = new UserForm();
        userForm.setId(userId);
        userForm.setUsername("test");
        userForm.setNickname("更新后的昵称");
        userForm.setMobile("13900139000");
        userForm.setGender(1);
        userForm.setStatus(1);
        userForm.setDeptId(1L);
        userForm.setRoleIds(Arrays.asList(3L)); // GUEST 角色

        // When: 执行更新操作
        boolean result = sysUserService.updateUser(userId, userForm);

        // Then: 验证结果
        assertTrue(result, "更新用户应该成功");
    }

    @Test
    @DisplayName("删除用户 - 成功")
    void testDeleteUsers_Success() {
        // Given: 准备要删除的用户ID
        String idsStr = "3";

        // When: 执行删除操作
        boolean result = sysUserService.deleteUsers(idsStr);

        // Then: 验证结果
        assertTrue(result, "删除用户应该成功");
    }

    @Test
    @DisplayName("删除用户 - 批量删除")
    void testDeleteUsers_Batch() {
        // Given: 准备要删除的多个用户ID
        String idsStr = "2,3";

        // When: 执行批量删除操作
        boolean result = sysUserService.deleteUsers(idsStr);

        // Then: 验证结果
        assertTrue(result, "批量删除用户应该成功");
    }

    @Test
    @DisplayName("修改用户密码 - 成功")
    void testUpdatePassword_Success() {
        // Given: 准备用户ID和新密码
        Long userId = 2L;
        String newPassword = "newPassword123";

        // When: 执行密码修改操作
        boolean result = sysUserService.updatePassword(userId, newPassword);

        // Then: 验证结果
        assertTrue(result, "修改密码应该成功");
    }

    @Test
    @DisplayName("根据用户名获取认证信息")
    void testGetUserAuthInfo() {
        // Given: 准备用户名
        String username = "admin";

        // When: 获取认证信息
        var authInfo = sysUserService.getUserAuthInfo(username);

        // Then: 验证结果
        assertNotNull(authInfo, "认证信息不应为空");
        assertEquals(username, authInfo.getUsername(), "用户名应该匹配");
        assertNotNull(authInfo.getRoles(), "角色集合不应为空");
        assertNotNull(authInfo.getPerms(), "权限集合不应为空");
        assertNotNull(authInfo.getDataScope(), "数据权限范围不应为空");
    }
}

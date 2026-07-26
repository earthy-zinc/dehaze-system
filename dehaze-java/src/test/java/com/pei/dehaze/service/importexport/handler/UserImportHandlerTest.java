package com.pei.dehaze.service.importexport.handler;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.enums.GenderEnum;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.model.entity.SysRole;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.entity.SysUserRole;
import com.pei.dehaze.service.SysRoleService;
import com.pei.dehaze.service.SysUserRoleService;
import com.pei.dehaze.service.SysUserService;
import com.pei.dehaze.service.importexport.model.ImportFieldConfig;
import com.pei.dehaze.service.importexport.model.ImportOptions;
import com.pei.dehaze.service.importexport.model.ImportResult;
import com.pei.dehaze.service.strategy.ProgressCallback;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.*;

/**
 * 用户导入处理器单元测试
 *
 * @author earthy-zinc
 * @since 2026-07-27
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("用户导入处理器测试")
@MockitoSettings(strictness = Strictness.LENIENT)
class UserImportHandlerTest {

    @Mock
    private SysUserService userService;

    @Mock
    private SysRoleService roleService;

    @Mock
    private SysUserRoleService userRoleService;

    @Mock
    private PasswordEncoder passwordEncoder;

    @InjectMocks
    private UserImportHandler handler;

    private ProgressCallback callback;

    @BeforeEach
    void setUp() {
        callback = new NoopCallback();
        when(passwordEncoder.encode(SystemConstants.DEFAULT_PASSWORD)).thenReturn("encoded-pwd");
        when(userService.count(any(LambdaQueryWrapper.class))).thenReturn(0L);
        when(userService.save(any(SysUser.class))).thenAnswer(inv -> {
            SysUser u = inv.getArgument(0);
            u.setId(System.currentTimeMillis());
            return true;
        });
    }

    @Test
    @DisplayName("getModule - 返回 user")
    void testGetModule() {
        assertEquals("user", handler.getModule());
    }

    @Test
    @DisplayName("getFieldConfigs - 返回 6 个字段, username/nickname/mobile 必填")
    void testGetFieldConfigs() {
        List<ImportFieldConfig> fields = handler.getFieldConfigs();

        assertEquals(6, fields.size());
        assertEquals("username", fields.get(0).getField());
        assertTrue(fields.get(0).isRequired());
        assertEquals(64, fields.get(0).getMaxLength());
        assertEquals("nickname", fields.get(1).getField());
        assertTrue(fields.get(1).isRequired());
        assertEquals("genderLabel", fields.get(2).getField());
        assertFalse(fields.get(2).isRequired());
        assertEquals("mobile", fields.get(3).getField());
        assertTrue(fields.get(3).isRequired());
        assertNotNull(fields.get(3).getRegex());
        assertEquals("email", fields.get(4).getField());
        assertNotNull(fields.get(4).getRegex());
        assertEquals("roleCodes", fields.get(5).getField());
    }

    @Test
    @DisplayName("getTemplateSampleData - 返回示例数据")
    void testGetTemplateSampleData() {
        List<Map<String, Object>> samples = handler.getTemplateSampleData();

        assertEquals(1, samples.size());
        Map<String, Object> sample = samples.get(0);
        assertEquals("zhangsan", sample.get("username"));
        assertEquals("张三", sample.get("nickname"));
        assertEquals("男", sample.get("genderLabel"));
        assertEquals("13800138000", sample.get("mobile"));
        assertEquals("zhangsan@example.com", sample.get("email"));
        assertEquals("user", sample.get("roleCodes"));
    }

    @Test
    @DisplayName("全量模式 - 全部成功(无角色)")
    void testImportBatch_AllMode_Success_NoRoles() {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("username", "lisi");
        row.put("nickname", "李四");
        row.put("genderLabel", "男");
        row.put("mobile", "13900139000");
        row.put("email", "lisi@example.com");

        ImportResult result = handler.importBatch(List.of(row),
                ImportOptions.of("all", null), callback);

        assertEquals(1, result.getTotalRows());
        assertEquals(1, result.getSuccessCount());
        assertEquals(0, result.getFailureCount());
        assertTrue(result.getErrors().isEmpty());

        ArgumentCaptor<SysUser> captor = ArgumentCaptor.forClass(SysUser.class);
        verify(userService).save(captor.capture());
        SysUser saved = captor.getValue();
        assertEquals("lisi", saved.getUsername());
        assertEquals("李四", saved.getNickname());
        assertEquals("13900139000", saved.getMobile());
        assertEquals("lisi@example.com", saved.getEmail());
        assertEquals("encoded-pwd", saved.getPassword());
        assertEquals(StatusEnum.ENABLE.getValue(), saved.getStatus());
        assertEquals(GenderEnum.MALE.getValue(), saved.getGender());
        verify(userRoleService, never()).saveBatch(anyList());
    }

    @Test
    @DisplayName("全量模式 - 带角色编码, 保存用户和角色关联")
    void testImportBatch_AllMode_WithRoles() {
        SysRole roleUser = new SysRole();
        roleUser.setId(10L);
        roleUser.setCode("user");
        SysRole roleAdmin = new SysRole();
        roleAdmin.setId(20L);
        roleAdmin.setCode("admin");
        when(roleService.list(any(LambdaQueryWrapper.class)))
                .thenReturn(List.of(roleUser, roleAdmin));

        Map<String, Object> row = new LinkedHashMap<>();
        row.put("username", "wangwu");
        row.put("nickname", "王五");
        row.put("mobile", "13700137000");
        row.put("roleCodes", "user,admin,unknown");

        ImportResult result = handler.importBatch(List.of(row),
                ImportOptions.of("all", Map.of("deptId", 100)), callback);

        assertEquals(1, result.getSuccessCount());

        ArgumentCaptor<SysUser> userCaptor = ArgumentCaptor.forClass(SysUser.class);
        verify(userService).save(userCaptor.capture());
        assertEquals(100L, userCaptor.getValue().getDeptId());

        ArgumentCaptor<List<SysUserRole>> rolesCaptor = ArgumentCaptor.forClass(List.class);
        verify(userRoleService).saveBatch(rolesCaptor.capture());
        List<SysUserRole> savedRoles = rolesCaptor.getValue();
        assertEquals(2, savedRoles.size());
        assertTrue(savedRoles.stream().anyMatch(r -> r.getRoleId() == 10L));
        assertTrue(savedRoles.stream().anyMatch(r -> r.getRoleId() == 20L));
    }

    @Test
    @DisplayName("全量模式 - 用户名已存在, 整体回滚抛异常")
    void testImportBatch_AllMode_DuplicateUsername_Throws() {
        when(userService.count(any(LambdaQueryWrapper.class))).thenReturn(1L);
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("username", "existing");
        row.put("nickname", "重复");
        row.put("mobile", "13800138000");

        RuntimeException ex = assertThrows(RuntimeException.class,
                () -> handler.importBatch(List.of(row), ImportOptions.of("all"), callback));
        assertTrue(ex.getMessage().contains("全量模式已回滚"));
        verify(userService, never()).save(any(SysUser.class));
    }

    @Test
    @DisplayName("部分模式 - 用户名已存在, 跳过错误行继续导入")
    void testImportBatch_PartialMode_DuplicateUsername_Skips() {
        when(userService.count(any(LambdaQueryWrapper.class)))
                .thenReturn(1L)
                .thenReturn(0L);

        Map<String, Object> badRow = new LinkedHashMap<>();
        badRow.put("username", "dup");
        badRow.put("nickname", "重复");
        badRow.put("mobile", "13800138000");
        Map<String, Object> goodRow = new LinkedHashMap<>();
        goodRow.put("username", "newuser");
        goodRow.put("nickname", "新用户");
        goodRow.put("mobile", "13900139000");

        ImportResult result = handler.importBatch(List.of(badRow, goodRow),
                ImportOptions.of("partial"), callback);

        assertEquals(2, result.getTotalRows());
        assertEquals(1, result.getSuccessCount());
        assertEquals(1, result.getFailureCount());
        assertEquals(1, result.getErrors().size());
        assertEquals(2, result.getErrors().get(0).getRow());
        assertTrue(result.getErrors().get(0).getMessage().contains("dup"));
        verify(userService, times(1)).save(any(SysUser.class));
    }

    @Test
    @DisplayName("全量模式 - 用户名为空, 抛异常")
    void testImportBatch_AllMode_BlankUsername_Throws() {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("username", "");
        row.put("nickname", "昵称");
        row.put("mobile", "13800138000");

        RuntimeException ex = assertThrows(RuntimeException.class,
                () -> handler.importBatch(List.of(row), ImportOptions.of("all"), callback));
        assertTrue(ex.getMessage().contains("用户名为空"));
    }

    @Test
    @DisplayName("全量模式 - 昵称为空, 抛异常")
    void testImportBatch_AllMode_BlankNickname_Throws() {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("username", "u1");
        row.put("nickname", "");
        row.put("mobile", "13800138000");

        RuntimeException ex = assertThrows(RuntimeException.class,
                () -> handler.importBatch(List.of(row), ImportOptions.of("all"), callback));
        assertTrue(ex.getMessage().contains("昵称为空"));
    }

    @Test
    @DisplayName("全量模式 - 手机号为空, 抛异常")
    void testImportBatch_AllMode_BlankMobile_Throws() {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("username", "u1");
        row.put("nickname", "n1");
        row.put("mobile", "");

        RuntimeException ex = assertThrows(RuntimeException.class,
                () -> handler.importBatch(List.of(row), ImportOptions.of("all"), callback));
        assertTrue(ex.getMessage().contains("手机号为空"));
    }

    @Test
    @DisplayName("全量模式 - 手机号格式不正确, 抛异常")
    void testImportBatch_AllMode_InvalidMobile_Throws() {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("username", "u1");
        row.put("nickname", "n1");
        row.put("mobile", "12345");

        RuntimeException ex = assertThrows(RuntimeException.class,
                () -> handler.importBatch(List.of(row), ImportOptions.of("all"), callback));
        assertTrue(ex.getMessage().contains("手机号格式不正确"));
    }

    @Test
    @DisplayName("全量模式 - 邮箱格式不正确, 抛异常")
    void testImportBatch_AllMode_InvalidEmail_Throws() {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("username", "u1");
        row.put("nickname", "n1");
        row.put("mobile", "13800138000");
        row.put("email", "not-an-email");

        RuntimeException ex = assertThrows(RuntimeException.class,
                () -> handler.importBatch(List.of(row), ImportOptions.of("all"), callback));
        assertTrue(ex.getMessage().contains("邮箱格式不正确"));
    }

    @Test
    @DisplayName("全量模式 - 性别取值无效, 抛异常")
    void testImportBatch_AllMode_InvalidGender_Throws() {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("username", "u1");
        row.put("nickname", "n1");
        row.put("mobile", "13800138000");
        row.put("genderLabel", "未知");

        RuntimeException ex = assertThrows(RuntimeException.class,
                () -> handler.importBatch(List.of(row), ImportOptions.of("all"), callback));
        assertTrue(ex.getMessage().contains("性别取值无效"));
    }

    @Test
    @DisplayName("全量模式 - 性别为女, 转换为 value=2")
    void testImportBatch_AllMode_FemaleGender() {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("username", "u1");
        row.put("nickname", "n1");
        row.put("mobile", "13800138000");
        row.put("genderLabel", "女");

        handler.importBatch(List.of(row), ImportOptions.of("all"), callback);

        ArgumentCaptor<SysUser> captor = ArgumentCaptor.forClass(SysUser.class);
        verify(userService).save(captor.capture());
        assertEquals(GenderEnum.FEMALE.getValue(), captor.getValue().getGender());
    }

    @Test
    @DisplayName("部分模式 - 保存失败时记录错误")
    void testImportBatch_PartialMode_SaveFails() {
        when(userService.save(any(SysUser.class))).thenReturn(false);
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("username", "u1");
        row.put("nickname", "n1");
        row.put("mobile", "13800138000");

        ImportResult result = handler.importBatch(List.of(row),
                ImportOptions.of("partial"), callback);

        assertEquals(1, result.getFailureCount());
        assertEquals(1, result.getErrors().size());
        assertEquals("保存失败", result.getErrors().get(0).getMessage());
    }

    @Test
    @DisplayName("全量模式 - 多行成功时累加 successCount")
    void testImportBatch_AllMode_MultipleRows() {
        Map<String, Object> row1 = new LinkedHashMap<>();
        row1.put("username", "u1");
        row1.put("nickname", "n1");
        row1.put("mobile", "13800138001");
        Map<String, Object> row2 = new LinkedHashMap<>();
        row2.put("username", "u2");
        row2.put("nickname", "n2");
        row2.put("mobile", "13800138002");

        ImportResult result = handler.importBatch(List.of(row1, row2),
                ImportOptions.of("all"), callback);

        assertEquals(2, result.getTotalRows());
        assertEquals(2, result.getSuccessCount());
        verify(userService, times(2)).save(any(SysUser.class));
    }

    @Test
    @DisplayName("部分模式 - 错误行号从 2 开始(表头占第 1 行)")
    void testImportBatch_PartialMode_RowNumberStartsAtTwo() {
        when(userService.count(any(LambdaQueryWrapper.class))).thenReturn(1L);
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("username", "dup");
        row.put("nickname", "n");
        row.put("mobile", "13800138000");

        ImportResult result = handler.importBatch(List.of(row),
                ImportOptions.of("partial"), callback);

        assertEquals(2, result.getErrors().get(0).getRow());
    }

    private static class NoopCallback implements ProgressCallback {
        @Override
        public void updateProgress(int current, int total, String message) {
        }

        @Override
        public boolean isCancelled() {
            return false;
        }
    }
}

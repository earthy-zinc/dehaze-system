package com.pei.dehaze.service.importexport.handler;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.model.entity.SysRole;
import com.pei.dehaze.service.SysRoleService;
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

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * 角色导入处理器单元测试
 *
 * @author earthy-zinc
 * @since 2026-07-27
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("角色导入处理器测试")
@MockitoSettings(strictness = Strictness.LENIENT)
class RoleImportHandlerTest {

    @Mock
    private SysRoleService roleService;

    @InjectMocks
    private RoleImportHandler handler;

    private ProgressCallback callback;

    @BeforeEach
    void setUp() {
        callback = new NoopCallback();
        when(roleService.count(any(LambdaQueryWrapper.class))).thenReturn(0L);
        when(roleService.save(any(SysRole.class))).thenAnswer(inv -> {
            SysRole r = inv.getArgument(0);
            r.setId(System.currentTimeMillis());
            return true;
        });
    }

    @Test
    @DisplayName("getModule - 返回 role")
    void testGetModule() {
        assertEquals("role", handler.getModule());
    }

    @Test
    @DisplayName("getFieldConfigs - 返回 4 个字段, name/code 必填")
    void testGetFieldConfigs() {
        List<ImportFieldConfig> fields = handler.getFieldConfigs();

        assertEquals(4, fields.size());
        assertEquals("name", fields.get(0).getField());
        assertTrue(fields.get(0).isRequired());
        assertEquals("code", fields.get(1).getField());
        assertTrue(fields.get(1).isRequired());
        assertEquals(64, fields.get(0).getMaxLength());
        assertEquals(32, fields.get(1).getMaxLength());
        assertEquals("sort", fields.get(2).getField());
        assertFalse(fields.get(2).isRequired());
        assertEquals("statusLabel", fields.get(3).getField());
    }

    @Test
    @DisplayName("getTemplateSampleData - 返回示例数据")
    void testGetTemplateSampleData() {
        List<Map<String, Object>> samples = handler.getTemplateSampleData();

        assertEquals(1, samples.size());
        Map<String, Object> sample = samples.get(0);
        assertEquals("普通用户", sample.get("name"));
        assertEquals("user", sample.get("code"));
        assertEquals("1", sample.get("sort"));
        assertEquals("启用", sample.get("statusLabel"));
    }

    @Test
    @DisplayName("全量模式 - 全部成功")
    void testImportBatch_AllMode_Success() {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("name", "测试角色");
        row.put("code", "test_role");
        row.put("sort", "5");
        row.put("statusLabel", "启用");

        ImportResult result = handler.importBatch(List.of(row), ImportOptions.of("all"), callback);

        assertEquals(1, result.getTotalRows());
        assertEquals(1, result.getSuccessCount());
        assertEquals(0, result.getFailureCount());
        assertTrue(result.getErrors().isEmpty());

        ArgumentCaptor<SysRole> captor = ArgumentCaptor.forClass(SysRole.class);
        verify(roleService).save(captor.capture());
        SysRole saved = captor.getValue();
        assertEquals("测试角色", saved.getName());
        assertEquals("test_role", saved.getCode());
        assertEquals(5, saved.getSort());
        assertEquals(StatusEnum.ENABLE.getValue(), saved.getStatus());
        assertEquals(5, saved.getDataScope());
    }

    @Test
    @DisplayName("全量模式 - 角色编码已存在, 整体回滚抛异常")
    void testImportBatch_AllMode_DuplicateCode_Throws() {
        when(roleService.count(any(LambdaQueryWrapper.class))).thenReturn(1L);
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("name", "重复角色");
        row.put("code", "existing_code");

        RuntimeException ex = assertThrows(RuntimeException.class,
                () -> handler.importBatch(List.of(row), ImportOptions.of("all"), callback));
        assertTrue(ex.getMessage().contains("全量模式已回滚"));
        verify(roleService, never()).save(any(SysRole.class));
    }

    @Test
    @DisplayName("部分模式 - 角色编码已存在, 跳过错误行继续导入")
    void testImportBatch_PartialMode_DuplicateCode_Skips() {
        when(roleService.count(any(LambdaQueryWrapper.class)))
                .thenReturn(1L)
                .thenReturn(0L);

        Map<String, Object> badRow = new LinkedHashMap<>();
        badRow.put("name", "重复角色");
        badRow.put("code", "dup_code");
        Map<String, Object> goodRow = new LinkedHashMap<>();
        goodRow.put("name", "新角色");
        goodRow.put("code", "new_code");

        ImportResult result = handler.importBatch(List.of(badRow, goodRow),
                ImportOptions.of("partial"), callback);

        assertEquals(2, result.getTotalRows());
        assertEquals(1, result.getSuccessCount());
        assertEquals(1, result.getFailureCount());
        assertEquals(1, result.getErrors().size());
        assertEquals(2, result.getErrors().get(0).getRow());
        assertTrue(result.getErrors().get(0).getMessage().contains("dup_code"));
        verify(roleService, times(1)).save(any(SysRole.class));
    }

    @Test
    @DisplayName("全量模式 - 名称为空, 抛异常")
    void testImportBatch_AllMode_BlankName_Throws() {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("name", "");
        row.put("code", "c1");

        RuntimeException ex = assertThrows(RuntimeException.class,
                () -> handler.importBatch(List.of(row), ImportOptions.of("all"), callback));
        assertTrue(ex.getMessage().contains("角色名称为空"));
    }

    @Test
    @DisplayName("全量模式 - 编码为空, 抛异常")
    void testImportBatch_AllMode_BlankCode_Throws() {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("name", "角色");
        row.put("code", "");

        RuntimeException ex = assertThrows(RuntimeException.class,
                () -> handler.importBatch(List.of(row), ImportOptions.of("all"), callback));
        assertTrue(ex.getMessage().contains("角色编码为空"));
    }

    @Test
    @DisplayName("部分模式 - 保存失败时记录错误")
    void testImportBatch_PartialMode_SaveFails() {
        when(roleService.save(any(SysRole.class))).thenReturn(false);
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("name", "角色");
        row.put("code", "c1");

        ImportResult result = handler.importBatch(List.of(row),
                ImportOptions.of("partial"), callback);

        assertEquals(1, result.getFailureCount());
        assertEquals(1, result.getErrors().size());
        assertEquals("保存失败", result.getErrors().get(0).getMessage());
    }

    @Test
    @DisplayName("状态字段解析 - 禁用")
    void testImportBatch_StatusDisabled() {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("name", "R");
        row.put("code", "c1");
        row.put("statusLabel", "禁用");

        handler.importBatch(List.of(row), ImportOptions.of("all"), callback);

        ArgumentCaptor<SysRole> captor = ArgumentCaptor.forClass(SysRole.class);
        verify(roleService).save(captor.capture());
        assertEquals(StatusEnum.DISABLE.getValue(), captor.getValue().getStatus());
    }

    @Test
    @DisplayName("状态字段解析 - 默认启用(空值)")
    void testImportBatch_StatusDefault() {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("name", "R");
        row.put("code", "c1");

        handler.importBatch(List.of(row), ImportOptions.of("all"), callback);

        ArgumentCaptor<SysRole> captor = ArgumentCaptor.forClass(SysRole.class);
        verify(roleService).save(captor.capture());
        assertEquals(StatusEnum.ENABLE.getValue(), captor.getValue().getStatus());
    }

    @Test
    @DisplayName("排序字段解析 - 非数字时取默认值 0")
    void testImportBatch_SortInvalid() {
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("name", "R");
        row.put("code", "c1");
        row.put("sort", "abc");

        handler.importBatch(List.of(row), ImportOptions.of("all"), callback);

        ArgumentCaptor<SysRole> captor = ArgumentCaptor.forClass(SysRole.class);
        verify(roleService).save(captor.capture());
        assertEquals(0, captor.getValue().getSort());
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

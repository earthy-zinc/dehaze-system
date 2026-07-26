package com.pei.dehaze.service.importexport.handler;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.model.query.RolePageQuery;
import com.pei.dehaze.model.vo.RolePageVO;
import com.pei.dehaze.service.SysRoleService;
import com.pei.dehaze.service.importexport.model.ExportContext;
import com.pei.dehaze.service.importexport.model.ExportDataProvider;
import com.pei.dehaze.service.importexport.model.ExportFieldConfig;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

/**
 * 角色导出处理器单元测试
 *
 * @author earthy-zinc
 * @since 2026-07-27
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("角色导出处理器测试")
@MockitoSettings(strictness = Strictness.LENIENT)
class RoleExportHandlerTest {

    @Mock
    private SysRoleService roleService;

    @InjectMocks
    private RoleExportHandler handler;

    private RolePageVO sampleRole;

    @BeforeEach
    void setUp() {
        sampleRole = new RolePageVO();
        sampleRole.setId(1L);
        sampleRole.setName("普通用户");
        sampleRole.setCode("user");
        sampleRole.setSort(1);
        sampleRole.setStatus(StatusEnum.ENABLE.getValue());
        sampleRole.setCreateTime(LocalDateTime.of(2026, 7, 27, 10, 0, 0));
    }

    @Test
    @DisplayName("getModule - 返回 role")
    void testGetModule() {
        assertEquals("role", handler.getModule());
    }

    @Test
    @DisplayName("getFieldConfigs - 返回 5 个字段且按 order 排序")
    void testGetFieldConfigs() {
        List<ExportFieldConfig> fields = handler.getFieldConfigs();

        assertEquals(5, fields.size());
        assertEquals("name", fields.get(0).getField());
        assertEquals("角色名称", fields.get(0).getLabel());
        assertEquals(1, fields.get(0).getOrder());
        assertEquals("code", fields.get(1).getField());
        assertEquals("sort", fields.get(2).getField());
        assertEquals("statusLabel", fields.get(3).getField());
        assertEquals("createTime", fields.get(4).getField());
        assertEquals("yyyy-MM-dd HH:mm:ss", fields.get(4).getDateFormat());
    }

    @Test
    @DisplayName("estimateCount - 调用 getRolePage 返回 total")
    void testEstimateCount() {
        Page<RolePageVO> page = new Page<>();
        page.setTotal(42L);
        when(roleService.getRolePage(any(RolePageQuery.class))).thenReturn(page);

        long count = handler.estimateCount(Map.of("keywords", "管理员"));

        assertEquals(42L, count);
    }

    @Test
    @DisplayName("estimateCount - 无查询参数时返回 total")
    void testEstimateCount_NoParams() {
        Page<RolePageVO> page = new Page<>();
        page.setTotal(0L);
        when(roleService.getRolePage(any(RolePageQuery.class))).thenReturn(page);

        long count = handler.estimateCount(null);

        assertEquals(0L, count);
    }

    @Test
    @DisplayName("getDataProvider - 第 1 页返回数据行,第 2 页返回空")
    void testGetDataProvider_FetchAndMap() {
        Page<RolePageVO> page1 = new Page<>();
        page1.setRecords(List.of(sampleRole));
        Page<RolePageVO> page2 = new Page<>();
        page2.setRecords(List.of());
        when(roleService.getRolePage(any(RolePageQuery.class)))
                .thenReturn(page1)
                .thenReturn(page2);

        ExportContext ctx = new ExportContext();
        ctx.setQueryParams(Map.of());
        ExportDataProvider provider = handler.getDataProvider(ctx);

        List<List<Object>> batch1 = provider.fetchBatch(1, 1000);
        assertEquals(1, batch1.size());
        List<Object> row = batch1.get(0);
        assertEquals("普通用户", row.get(0));
        assertEquals("user", row.get(1));
        assertEquals(1, row.get(2));
        assertEquals("启用", row.get(3));
        assertEquals("2026-07-27 10:00:00", row.get(4));

        List<List<Object>> batch2 = provider.fetchBatch(2, 1000);
        assertTrue(batch2.isEmpty());
    }

    @Test
    @DisplayName("getDataProvider - selectedFields 过滤后只返回选中字段")
    void testGetDataProvider_SelectedFields() {
        Page<RolePageVO> page = new Page<>();
        page.setRecords(List.of(sampleRole));
        when(roleService.getRolePage(any(RolePageQuery.class))).thenReturn(page);

        ExportContext ctx = new ExportContext();
        ctx.setQueryParams(Map.of());
        ctx.setSelectedFields(List.of("name", "code"));
        ExportDataProvider provider = handler.getDataProvider(ctx);

        List<List<Object>> batch = provider.fetchBatch(1, 1000);
        assertEquals(1, batch.size());
        assertEquals(2, batch.get(0).size());
        assertEquals("普通用户", batch.get(0).get(0));
        assertEquals("user", batch.get(0).get(1));
    }

    @Test
    @DisplayName("getDataProvider - 空记录返回空列表")
    void testGetDataProvider_EmptyRecords() {
        Page<RolePageVO> page = new Page<>();
        page.setRecords(null);
        when(roleService.getRolePage(any(RolePageQuery.class))).thenReturn(page);

        ExportContext ctx = new ExportContext();
        ctx.setQueryParams(Map.of());
        ExportDataProvider provider = handler.getDataProvider(ctx);

        assertTrue(provider.fetchBatch(1, 1000).isEmpty());
    }

    @Test
    @DisplayName("getDataProvider - status 为 null 时输出空字符串")
    void testGetDataProvider_NullStatus() {
        RolePageVO role = new RolePageVO();
        role.setName("R");
        role.setCode("r");
        role.setSort(0);
        role.setStatus(null);
        role.setCreateTime(null);
        Page<RolePageVO> page = new Page<>();
        page.setRecords(List.of(role));
        when(roleService.getRolePage(any(RolePageQuery.class))).thenReturn(page);

        ExportContext ctx = new ExportContext();
        ctx.setQueryParams(Map.of());
        ExportDataProvider provider = handler.getDataProvider(ctx);

        List<List<Object>> batch = provider.fetchBatch(1, 1000);
        assertEquals("", batch.get(0).get(3));
        assertEquals("", batch.get(0).get(4));
    }

    @Test
    @DisplayName("getDataProvider - 禁用状态输出 禁用")
    void testGetDataProvider_DisabledStatus() {
        sampleRole.setStatus(StatusEnum.DISABLE.getValue());
        Page<RolePageVO> page = new Page<>();
        page.setRecords(List.of(sampleRole));
        when(roleService.getRolePage(any(RolePageQuery.class))).thenReturn(page);

        ExportContext ctx = new ExportContext();
        ctx.setQueryParams(Map.of());
        ExportDataProvider provider = handler.getDataProvider(ctx);

        List<List<Object>> batch = provider.fetchBatch(1, 1000);
        assertEquals("禁用", batch.get(0).get(3));
    }

    @Test
    @DisplayName("export 方法为空实现 - 不抛异常")
    void testExport_NoOp() throws Exception {
        assertDoesNotThrow(() -> handler.export(new ExportContext(), null));
    }
}

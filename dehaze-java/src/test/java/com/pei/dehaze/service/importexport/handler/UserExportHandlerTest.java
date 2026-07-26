package com.pei.dehaze.service.importexport.handler;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.model.query.UserPageQuery;
import com.pei.dehaze.model.vo.UserPageVO;
import com.pei.dehaze.service.SysUserService;
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

import java.util.Date;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

/**
 * 用户导出处理器单元测试
 *
 * @author earthy-zinc
 * @since 2026-07-27
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("用户导出处理器测试")
@MockitoSettings(strictness = Strictness.LENIENT)
class UserExportHandlerTest {

    @Mock
    private SysUserService userService;

    @InjectMocks
    private UserExportHandler handler;

    private UserPageVO sampleUser;

    @BeforeEach
    void setUp() {
        sampleUser = new UserPageVO();
        sampleUser.setId(1L);
        sampleUser.setUsername("zhangsan");
        sampleUser.setNickname("张三");
        sampleUser.setDeptName("研发部");
        sampleUser.setGenderLabel("男");
        sampleUser.setMobile("13800138000");
        sampleUser.setEmail("zhangsan@example.com");
        sampleUser.setStatus(StatusEnum.ENABLE.getValue());
        sampleUser.setCreateTime(new Date(1735000000000L));
    }

    @Test
    @DisplayName("getModule - 返回 user")
    void testGetModule() {
        assertEquals("user", handler.getModule());
    }

    @Test
    @DisplayName("getFieldConfigs - 返回 8 个字段且按 order 排序")
    void testGetFieldConfigs() {
        List<ExportFieldConfig> fields = handler.getFieldConfigs();

        assertEquals(8, fields.size());
        assertEquals("username", fields.get(0).getField());
        assertEquals("用户名", fields.get(0).getLabel());
        assertEquals(1, fields.get(0).getOrder());
        assertEquals("nickname", fields.get(1).getField());
        assertEquals("deptName", fields.get(2).getField());
        assertEquals("genderLabel", fields.get(3).getField());
        assertEquals("mobile", fields.get(4).getField());
        assertEquals("email", fields.get(5).getField());
        assertEquals("statusLabel", fields.get(6).getField());
        assertEquals("createTime", fields.get(7).getField());
        assertEquals("yyyy-MM-dd HH:mm:ss", fields.get(7).getDateFormat());
    }

    @Test
    @DisplayName("estimateCount - 调用 listPagedUsers 返回 total")
    void testEstimateCount() {
        Page<UserPageVO> page = new Page<>();
        page.setTotal(123L);
        when(userService.listPagedUsers(any(UserPageQuery.class))).thenReturn(page);

        long count = handler.estimateCount(Map.of("keywords", "张", "status", 1, "deptId", 100));

        assertEquals(123L, count);
    }

    @Test
    @DisplayName("estimateCount - 无参数时返回 total")
    void testEstimateCount_NullParams() {
        Page<UserPageVO> page = new Page<>();
        page.setTotal(0L);
        when(userService.listPagedUsers(any(UserPageQuery.class))).thenReturn(page);

        assertEquals(0L, handler.estimateCount(null));
    }

    @Test
    @DisplayName("estimateCount - 空字符串参数被忽略不报错")
    void testEstimateCount_EmptyStringParams() {
        Page<UserPageVO> page = new Page<>();
        page.setTotal(0L);
        when(userService.listPagedUsers(any(UserPageQuery.class))).thenReturn(page);

        Map<String, Object> params = new java.util.HashMap<>();
        params.put("status", "");
        params.put("deptId", "");
        params.put("startTime", "2026-01-01");
        params.put("endTime", "2026-12-31");

        assertEquals(0L, handler.estimateCount(params));
    }

    @Test
    @DisplayName("getDataProvider - 第 1 页返回数据行,第 2 页返回空")
    void testGetDataProvider_FetchAndMap() {
        Page<UserPageVO> page1 = new Page<>();
        page1.setRecords(List.of(sampleUser));
        Page<UserPageVO> page2 = new Page<>();
        page2.setRecords(List.of());
        when(userService.listPagedUsers(any(UserPageQuery.class)))
                .thenReturn(page1)
                .thenReturn(page2);

        ExportContext ctx = new ExportContext();
        ctx.setQueryParams(Map.of());
        ExportDataProvider provider = handler.getDataProvider(ctx);

        List<List<Object>> batch1 = provider.fetchBatch(1, 1000);
        assertEquals(1, batch1.size());
        List<Object> row = batch1.get(0);
        assertEquals("zhangsan", row.get(0));
        assertEquals("张三", row.get(1));
        assertEquals("研发部", row.get(2));
        assertEquals("男", row.get(3));
        assertEquals("13800138000", row.get(4));
        assertEquals("zhangsan@example.com", row.get(5));
        assertEquals("启用", row.get(6));
        assertTrue(((String) row.get(7)).matches("\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}"));

        List<List<Object>> batch2 = provider.fetchBatch(2, 1000);
        assertTrue(batch2.isEmpty());
    }

    @Test
    @DisplayName("getDataProvider - selectedFields 过滤后只返回选中字段")
    void testGetDataProvider_SelectedFields() {
        Page<UserPageVO> page = new Page<>();
        page.setRecords(List.of(sampleUser));
        when(userService.listPagedUsers(any(UserPageQuery.class))).thenReturn(page);

        ExportContext ctx = new ExportContext();
        ctx.setQueryParams(Map.of());
        ctx.setSelectedFields(List.of("username", "mobile"));
        ExportDataProvider provider = handler.getDataProvider(ctx);

        List<List<Object>> batch = provider.fetchBatch(1, 1000);
        assertEquals(1, batch.size());
        assertEquals(2, batch.get(0).size());
        assertEquals("zhangsan", batch.get(0).get(0));
        assertEquals("13800138000", batch.get(0).get(1));
    }

    @Test
    @DisplayName("getDataProvider - 空记录返回空列表")
    void testGetDataProvider_EmptyRecords() {
        Page<UserPageVO> page = new Page<>();
        page.setRecords(null);
        when(userService.listPagedUsers(any(UserPageQuery.class))).thenReturn(page);

        ExportContext ctx = new ExportContext();
        ctx.setQueryParams(Map.of());
        ExportDataProvider provider = handler.getDataProvider(ctx);

        assertTrue(provider.fetchBatch(1, 1000).isEmpty());
    }

    @Test
    @DisplayName("getDataProvider - status 为 null 时输出空字符串")
    void testGetDataProvider_NullStatus() {
        UserPageVO user = new UserPageVO();
        user.setUsername("u");
        user.setStatus(null);
        user.setCreateTime(null);
        Page<UserPageVO> page = new Page<>();
        page.setRecords(List.of(user));
        when(userService.listPagedUsers(any(UserPageQuery.class))).thenReturn(page);

        ExportContext ctx = new ExportContext();
        ctx.setQueryParams(Map.of());
        ExportDataProvider provider = handler.getDataProvider(ctx);

        List<List<Object>> batch = provider.fetchBatch(1, 1000);
        assertEquals("", batch.get(0).get(6));
        assertEquals("", batch.get(0).get(7));
    }

    @Test
    @DisplayName("getDataProvider - 禁用状态输出 禁用")
    void testGetDataProvider_DisabledStatus() {
        sampleUser.setStatus(StatusEnum.DISABLE.getValue());
        Page<UserPageVO> page = new Page<>();
        page.setRecords(List.of(sampleUser));
        when(userService.listPagedUsers(any(UserPageQuery.class))).thenReturn(page);

        ExportContext ctx = new ExportContext();
        ctx.setQueryParams(Map.of());
        ExportDataProvider provider = handler.getDataProvider(ctx);

        List<List<Object>> batch = provider.fetchBatch(1, 1000);
        assertEquals("禁用", batch.get(0).get(6));
    }

    @Test
    @DisplayName("getDataProvider - 字符串字段为 null 时输出空字符串")
    void testGetDataProvider_NullStringFields() {
        UserPageVO user = new UserPageVO();
        user.setUsername(null);
        user.setNickname(null);
        user.setDeptName(null);
        user.setGenderLabel(null);
        user.setMobile(null);
        user.setEmail(null);
        user.setStatus(StatusEnum.ENABLE.getValue());
        Page<UserPageVO> page = new Page<>();
        page.setRecords(List.of(user));
        when(userService.listPagedUsers(any(UserPageQuery.class))).thenReturn(page);

        ExportContext ctx = new ExportContext();
        ctx.setQueryParams(Map.of());
        ExportDataProvider provider = handler.getDataProvider(ctx);

        List<List<Object>> batch = provider.fetchBatch(1, 1000);
        for (int i = 0; i < 6; i++) {
            assertEquals("", batch.get(0).get(i));
        }
    }

    @Test
    @DisplayName("export 方法为空实现 - 不抛异常")
    void testExport_NoOp() throws Exception {
        assertDoesNotThrow(() -> handler.export(new ExportContext(), null));
    }
}

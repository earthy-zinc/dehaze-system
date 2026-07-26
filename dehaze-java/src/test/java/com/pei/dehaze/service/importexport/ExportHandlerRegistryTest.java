package com.pei.dehaze.service.importexport;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.service.importexport.model.ExportContext;
import com.pei.dehaze.service.importexport.model.ExportDataProvider;
import com.pei.dehaze.service.importexport.model.ExportFieldConfig;
import com.pei.dehaze.service.strategy.ProgressCallback;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 导出处理器注册表单元测试
 *
 * @author earthy-zinc
 * @since 2026-07-27
 */
@DisplayName("导出处理器注册表测试")
class ExportHandlerRegistryTest {

    @Test
    @DisplayName("构造时按 module 注册 handler")
    void testConstructor_RegistersHandlersByModule() {
        ExportHandler userHandler = new StubExportHandler("user");
        ExportHandler roleHandler = new StubExportHandler("role");
        ExportHandlerRegistry registry = new ExportHandlerRegistry(List.of(userHandler, roleHandler));

        assertSame(userHandler, registry.getHandler("user"));
        assertSame(roleHandler, registry.getHandler("role"));
    }

    @Test
    @DisplayName("supports - 已注册返回 true,未注册返回 false")
    void testSupports() {
        ExportHandlerRegistry registry = new ExportHandlerRegistry(
                List.of(new StubExportHandler("user")));

        assertTrue(registry.supports("user"));
        assertFalse(registry.supports("role"));
    }

    @Test
    @DisplayName("getHandler - 未注册模块抛 A0710")
    void testGetHandler_NotRegistered() {
        ExportHandlerRegistry registry = new ExportHandlerRegistry(List.of());

        BusinessException ex = assertThrows(BusinessException.class,
                () -> registry.getHandler("user"));
        assertEquals(ResultCode.MODULE_IMPORT_NOT_SUPPORTED.getCode(), ex.getResultCode().getCode());
    }

    @Test
    @DisplayName("构造时重复注册同一 module 抛 IllegalStateException")
    void testConstructor_DuplicateModuleThrows() {
        ExportHandler h1 = new StubExportHandler("user");
        ExportHandler h2 = new StubExportHandler("user");

        assertThrows(IllegalStateException.class,
                () -> new ExportHandlerRegistry(List.of(h1, h2)));
    }

    @Test
    @DisplayName("空 handler 列表构造无异常")
    void testConstructor_EmptyList() {
        ExportHandlerRegistry registry = new ExportHandlerRegistry(List.of());
        assertFalse(registry.supports("user"));
    }

    private static class StubExportHandler implements ExportHandler {
        private final String module;

        StubExportHandler(String module) {
            this.module = module;
        }

        @Override
        public String getModule() {
            return module;
        }

        @Override
        public long estimateCount(Map<String, Object> queryParams) {
            return 0;
        }

        @Override
        public void export(ExportContext ctx, ProgressCallback callback) {
        }

        @Override
        public List<ExportFieldConfig> getFieldConfigs() {
            return List.of();
        }

        @Override
        public ExportDataProvider getDataProvider(ExportContext ctx) {
            return (pageNum, pageSize) -> List.of();
        }
    }
}

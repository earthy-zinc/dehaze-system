package com.pei.dehaze.service.importexport;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.service.importexport.model.ImportFieldConfig;
import com.pei.dehaze.service.importexport.model.ImportOptions;
import com.pei.dehaze.service.importexport.model.ImportResult;
import com.pei.dehaze.service.strategy.ProgressCallback;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 导入处理器注册表单元测试
 *
 * @author earthy-zinc
 * @since 2026-07-27
 */
@DisplayName("导入处理器注册表测试")
class ImportHandlerRegistryTest {

    @Test
    @DisplayName("构造时按 module 注册 handler")
    void testConstructor_RegistersHandlersByModule() {
        ImportHandler userHandler = new StubImportHandler("user");
        ImportHandler roleHandler = new StubImportHandler("role");
        ImportHandlerRegistry registry = new ImportHandlerRegistry(List.of(userHandler, roleHandler));

        assertSame(userHandler, registry.getHandler("user"));
        assertSame(roleHandler, registry.getHandler("role"));
    }

    @Test
    @DisplayName("supports - 已注册返回 true,未注册返回 false")
    void testSupports() {
        ImportHandlerRegistry registry = new ImportHandlerRegistry(
                List.of(new StubImportHandler("user")));

        assertTrue(registry.supports("user"));
        assertFalse(registry.supports("role"));
    }

    @Test
    @DisplayName("getHandler - 未注册模块抛 A0710")
    void testGetHandler_NotRegistered() {
        ImportHandlerRegistry registry = new ImportHandlerRegistry(List.of());

        BusinessException ex = assertThrows(BusinessException.class,
                () -> registry.getHandler("user"));
        assertEquals(ResultCode.MODULE_IMPORT_NOT_SUPPORTED.getCode(), ex.getResultCode().getCode());
    }

    @Test
    @DisplayName("构造时重复注册同一 module 抛 IllegalStateException")
    void testConstructor_DuplicateModuleThrows() {
        ImportHandler h1 = new StubImportHandler("user");
        ImportHandler h2 = new StubImportHandler("user");

        assertThrows(IllegalStateException.class,
                () -> new ImportHandlerRegistry(List.of(h1, h2)));
    }

    private static class StubImportHandler implements ImportHandler {
        private final String module;

        StubImportHandler(String module) {
            this.module = module;
        }

        @Override
        public String getModule() {
            return module;
        }

        @Override
        public List<ImportFieldConfig> getFieldConfigs() {
            return List.of();
        }

        @Override
        public ImportResult importBatch(List<Map<String, Object>> rows, ImportOptions options, ProgressCallback callback) {
            return ImportResult.success(0, 0);
        }
    }
}

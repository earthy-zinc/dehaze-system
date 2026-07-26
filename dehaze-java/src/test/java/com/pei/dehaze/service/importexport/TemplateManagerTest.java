package com.pei.dehaze.service.importexport;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.service.importexport.model.ImportFieldConfig;
import com.pei.dehaze.service.importexport.model.ImportOptions;
import com.pei.dehaze.service.importexport.model.ImportResult;
import com.pei.dehaze.service.strategy.ProgressCallback;
import jakarta.servlet.ServletOutputStream;
import jakarta.servlet.http.HttpServletResponse;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * 导入模板管理器单元测试
 *
 * @author earthy-zinc
 * @since 2026-07-27
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("导入模板管理器测试")
@MockitoSettings(strictness = Strictness.LENIENT)
class TemplateManagerTest {

    @Mock
    private ImportExportFileGenerator fileGenerator;

    @Mock
    private ImportHandlerRegistry importHandlerRegistry;

    @Mock
    private HttpServletResponse response;

    @Mock
    private ServletOutputStream servletOutputStream;

    private TemplateManager templateManager;

    @BeforeEach
    void setUp() throws IOException {
        templateManager = new TemplateManager(fileGenerator, importHandlerRegistry);
        when(response.getOutputStream()).thenReturn(servletOutputStream);
    }

    @Test
    @DisplayName("downloadTemplate - excel 格式调用 writeTemplateExcel")
    void testDownloadTemplate_Excel() throws IOException {
        ImportHandler handler = new StubImportHandler(
                List.of(ImportFieldConfig.of("username", "用户名", true)),
                List.of(Map.of("username", "zhangsan")));
        when(importHandlerRegistry.getHandler("user")).thenReturn(handler);

        templateManager.downloadTemplate("user", "excel", response);

        verify(fileGenerator).writeTemplateExcel(any(), anyList(), anyList());
        verify(response).setContentType("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
        verify(response).setHeader(eq("Content-Disposition"), contains("user_import_template.xlsx"));
    }

    @Test
    @DisplayName("downloadTemplate - csv 格式调用 writeTemplateCsv")
    void testDownloadTemplate_Csv() throws IOException {
        ImportHandler handler = new StubImportHandler(
                List.of(ImportFieldConfig.of("username", "用户名", true)),
                List.of());
        when(importHandlerRegistry.getHandler("user")).thenReturn(handler);

        templateManager.downloadTemplate("user", "csv", response);

        verify(fileGenerator).writeTemplateCsv(any(), anyList(), anyList());
        verify(response).setContentType("text/csv");
        verify(response).setHeader(eq("Content-Disposition"), contains("user_import_template.csv"));
    }

    @Test
    @DisplayName("downloadTemplate - null 格式默认走 excel")
    void testDownloadTemplate_NullFormatDefaultsExcel() throws IOException {
        ImportHandler handler = new StubImportHandler(
                List.of(ImportFieldConfig.of("username", "用户名", true)),
                List.of());
        when(importHandlerRegistry.getHandler("user")).thenReturn(handler);

        templateManager.downloadTemplate("user", null, response);

        verify(fileGenerator).writeTemplateExcel(any(), anyList(), anyList());
    }

    @Test
    @DisplayName("downloadTemplate - handler 不存在抛 A0710")
    void testDownloadTemplate_ModuleNotSupported() {
        when(importHandlerRegistry.getHandler("unknown"))
                .thenThrow(new BusinessException(ResultCode.MODULE_IMPORT_NOT_SUPPORTED, "模块 unknown 不支持导入"));

        BusinessException ex = assertThrows(BusinessException.class,
                () -> templateManager.downloadTemplate("unknown", "excel", response));
        assertEquals(ResultCode.MODULE_IMPORT_NOT_SUPPORTED.getCode(), ex.getResultCode().getCode());
    }

    @Test
    @DisplayName("downloadTemplate - 字段配置为空抛 A0710")
    void testDownloadTemplate_EmptyFields() {
        ImportHandler handler = new StubImportHandler(List.of(), List.of());
        when(importHandlerRegistry.getHandler("user")).thenReturn(handler);

        BusinessException ex = assertThrows(BusinessException.class,
                () -> templateManager.downloadTemplate("user", "excel", response));
        assertEquals(ResultCode.MODULE_IMPORT_NOT_SUPPORTED.getCode(), ex.getResultCode().getCode());
    }

    @Test
    @DisplayName("downloadTemplate - 生成器异常包装为 BusinessException")
    void testDownloadTemplate_GeneratorThrows() throws IOException {
        ImportHandler handler = new StubImportHandler(
                List.of(ImportFieldConfig.of("username", "用户名", true)),
                List.of());
        when(importHandlerRegistry.getHandler("user")).thenReturn(handler);
        doThrow(new IOException("磁盘已满"))
                .when(fileGenerator).writeTemplateCsv(any(), anyList(), anyList());

        BusinessException ex = assertThrows(BusinessException.class,
                () -> templateManager.downloadTemplate("user", "csv", response));
        assertEquals(ResultCode.SYSTEM_EXECUTION_ERROR.getCode(), ex.getResultCode().getCode());
    }

    private static class StubImportHandler implements ImportHandler {
        private final List<ImportFieldConfig> fields;
        private final List<Map<String, Object>> sampleData;

        StubImportHandler(List<ImportFieldConfig> fields, List<Map<String, Object>> sampleData) {
            this.fields = fields;
            this.sampleData = sampleData;
        }

        @Override
        public String getModule() {
            return "user";
        }

        @Override
        public List<ImportFieldConfig> getFieldConfigs() {
            return fields;
        }

        @Override
        public ImportResult importBatch(List<Map<String, Object>> rows, ImportOptions options, ProgressCallback callback) {
            return ImportResult.success(0, 0);
        }

        @Override
        public List<Map<String, Object>> getTemplateSampleData() {
            return sampleData;
        }
    }
}

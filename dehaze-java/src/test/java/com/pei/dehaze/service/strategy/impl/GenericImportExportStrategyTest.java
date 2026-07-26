package com.pei.dehaze.service.strategy.impl;

import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.importexport.ExportHandler;
import com.pei.dehaze.service.importexport.ExportHandlerRegistry;
import com.pei.dehaze.service.importexport.ImportExportFileGenerator;
import com.pei.dehaze.service.importexport.ImportHandler;
import com.pei.dehaze.service.importexport.ImportHandlerRegistry;
import com.pei.dehaze.service.importexport.model.ExportDataProvider;
import com.pei.dehaze.service.importexport.model.ExportFieldConfig;
import com.pei.dehaze.service.importexport.model.ImportFieldConfig;
import com.pei.dehaze.service.importexport.model.ImportOptions;
import com.pei.dehaze.service.importexport.model.ImportResult;
import com.pei.dehaze.service.strategy.ProgressCallback;
import com.pei.dehaze.service.strategy.TaskResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * 通用导入导出策略单元测试
 * <p>验证 {@link GenericExportStrategy} / {@link GenericImportStrategy} 的执行逻辑：
 * 成功时返回携带结果数据的 {@link TaskResult}，失败时返回错误信息。
 *
 * @author earthy-zinc
 * @since 2026-07-27
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("通用导入导出策略测试")
@MockitoSettings(strictness = Strictness.LENIENT)
class GenericImportExportStrategyTest {

    @Mock
    private ExportHandlerRegistry exportHandlerRegistry;
    @Mock
    private ImportHandlerRegistry importHandlerRegistry;
    @Mock
    private ImportExportFileGenerator fileGenerator;
    @Mock
    private FileService fileService;
    @Mock
    private ExportHandler exportHandler;
    @Mock
    private ImportHandler importHandler;

    private GenericExportStrategy exportStrategy;
    private GenericImportStrategy importStrategy;

    @BeforeEach
    void setUp() throws IOException {
        exportStrategy = new GenericExportStrategy(exportHandlerRegistry, fileGenerator, fileService);
        importStrategy = new GenericImportStrategy(importHandlerRegistry, fileGenerator, fileService);

        when(exportHandlerRegistry.getHandler("user")).thenReturn(exportHandler);
        when(importHandlerRegistry.getHandler("user")).thenReturn(importHandler);
        when(exportHandler.getModule()).thenReturn("user");
        when(exportHandler.useDirectExport()).thenReturn(false);
        when(exportHandler.getFieldConfigs()).thenReturn(List.of(
                ExportFieldConfig.of("username", "用户名", 1)
        ));
        when(importHandler.getDynamicFieldConfigs()).thenReturn(List.of(
                ImportFieldConfig.of("username", "用户名", true)
        ));
        mockFileGeneratorParse(0);
    }

    // ==================== 导出策略 ====================

    @Test
    @DisplayName("GenericExportStrategy - 完成后上传文件并返回下载 URL")
    void testExport_Success() throws IOException {
        SysTask task = new SysTask();
        task.setTaskId("task-export-001");
        when(exportHandler.estimateCount(anyMap())).thenReturn(10L);
        ExportDataProvider provider = (pageNum, pageSize) -> pageNum == 1
                ? List.of(List.of("user1"), List.of("user2"))
                : List.of();
        when(exportHandler.getDataProvider(any())).thenReturn(provider);
        when(fileService.uploadFile(anyString(), any(InputStream.class), anyLong(), anyString()))
                .thenReturn("http://minio/exports/task-export-001.xlsx");

        TaskResult result = exportStrategy.execute(task, Map.of(
                "module", "user",
                "format", "excel",
                "fields", List.of("username"),
                "query", Map.of()
        ), new NoopCallback());

        assertTrue(result.isSuccess());
        assertEquals("http://minio/exports/task-export-001.xlsx", result.getData());
        ArgumentCaptor<String> objectNameCaptor = ArgumentCaptor.forClass(String.class);
        verify(fileService).uploadFile(objectNameCaptor.capture(), any(InputStream.class), anyLong(), anyString());
        assertTrue(objectNameCaptor.getValue().startsWith("exports/task-export-001."));
    }

    @Test
    @DisplayName("GenericExportStrategy - 异常时返回失败结果")
    void testExport_Failure() throws IOException {
        SysTask task = new SysTask();
        task.setTaskId("task-export-fail");
        when(exportHandler.estimateCount(anyMap())).thenThrow(new RuntimeException("DB 超时"));

        TaskResult result = exportStrategy.execute(task, Map.of(
                "module", "user", "format", "excel"
        ), new NoopCallback());

        assertFalse(result.isSuccess());
        assertTrue(result.getErrorMessage().contains("导出失败"));
        verify(fileService, never()).uploadFile(anyString(), any(InputStream.class), anyLong(), anyString());
    }

    // ==================== 导入策略 ====================

    @Test
    @DisplayName("GenericImportStrategy - 完成后返回结果 JSON")
    void testImport_Success() throws IOException {
        SysTask task = new SysTask();
        task.setTaskId("task-import-001");
        String csvContent = "username\nuser1\nuser2\n";
        when(fileService.downLoadFile(anyString()))
                .thenReturn(new ByteArrayInputStream(csvContent.getBytes(StandardCharsets.UTF_8)));
        mockFileGeneratorParse(2);
        ImportResult importResult = ImportResult.success(2, 2);
        when(importHandler.importBatch(anyList(), any(ImportOptions.class), any())).thenReturn(importResult);

        TaskResult result = importStrategy.execute(task, Map.of(
                "module", "user",
                "fileObjectName", "temp/imports/abc.csv",
                "mode", "all"
        ), new NoopCallback());

        assertTrue(result.isSuccess());
        assertNotNull(result.getData());
    }

    @Test
    @DisplayName("GenericImportStrategy - 有失败行时生成错误报告")
    void testImport_WithFailures() throws IOException {
        SysTask task = new SysTask();
        task.setTaskId("task-import-002");
        String csvContent = "username\nuser1\n";
        when(fileService.downLoadFile(anyString()))
                .thenReturn(new ByteArrayInputStream(csvContent.getBytes(StandardCharsets.UTF_8)));
        mockFileGeneratorParse(1);
        ImportResult importResult = ImportResult.partial(1, 0, 1,
                List.of(ImportResult.ImportError.builder().row(2).message("用户名已存在").build()));
        when(importHandler.importBatch(anyList(), any(ImportOptions.class), any())).thenReturn(importResult);
        when(fileService.uploadFile(anyString(), any(InputStream.class), anyLong(), anyString()))
                .thenReturn("http://minio/imports/errors.xlsx");

        TaskResult result = importStrategy.execute(task, Map.of(
                "module", "user",
                "fileObjectName", "temp/imports/abc.csv",
                "mode", "partial"
        ), new NoopCallback());

        assertTrue(result.isSuccess());
        verify(fileService).uploadFile(contains("imports/"), any(InputStream.class), anyLong(), anyString());
    }

    @Test
    @DisplayName("GenericImportStrategy - 异常时返回失败结果")
    void testImport_Failure() throws IOException {
        SysTask task = new SysTask();
        task.setTaskId("task-import-fail");
        when(fileService.downLoadFile(anyString())).thenThrow(new RuntimeException("MinIO 不可用"));

        TaskResult result = importStrategy.execute(task, Map.of(
                "module", "user",
                "fileObjectName", "temp/imports/abc.csv",
                "mode", "all"
        ), new NoopCallback());

        assertFalse(result.isSuccess());
        assertTrue(result.getErrorMessage().contains("导入失败"));
    }

    // ==================== 工具方法 ====================

    private void mockFileGeneratorParse(int rowCount) throws IOException {
        doAnswer(invocation -> {
            ImportExportFileGenerator.RowConsumer consumer = invocation.getArgument(3);
            for (int i = 1; i <= rowCount; i++) {
                consumer.consume(i, Map.of("username", "user" + i));
            }
            return null;
        }).when(fileGenerator).parse(any(InputStream.class), anyString(), anyList(), any());
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

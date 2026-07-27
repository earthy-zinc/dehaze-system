package com.pei.dehaze.service.importexport;

import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.model.form.ExportTaskCreateForm;
import com.pei.dehaze.model.vo.ExportTaskVO;
import com.pei.dehaze.model.vo.ImportResultVO;
import com.pei.dehaze.model.vo.ImportTaskVO;
import com.pei.dehaze.model.vo.TaskVO;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.TaskService;
import com.pei.dehaze.service.importexport.model.ExportDataProvider;
import com.pei.dehaze.service.importexport.model.ExportFieldConfig;
import com.pei.dehaze.service.importexport.model.ImportFieldConfig;
import com.pei.dehaze.service.importexport.model.ImportOptions;
import com.pei.dehaze.service.importexport.model.ImportResult;
import com.pei.dehaze.service.importexport.model.ExportContext;
import com.pei.dehaze.service.strategy.ProgressCallback;
import jakarta.servlet.ServletOutputStream;
import jakarta.servlet.http.HttpServletResponse;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.multipart.MultipartFile;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * 通用导入导出服务单元测试
 * <p>覆盖：同步/异步判断、文件验证、Handler 路由、导出/导入任务创建。
 *
 * @author earthy-zinc
 * @since 2026-07-27
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("通用导入导出服务测试")
@MockitoSettings(strictness = Strictness.LENIENT)
class ImportExportServiceTest {

    @Mock
    private ExportHandlerRegistry exportHandlerRegistry;

    @Mock
    private ImportHandlerRegistry importHandlerRegistry;

    @Mock
    private ImportExportFileGenerator fileGenerator;

    @Mock
    private TaskService taskService;

    @Mock
    private FileService fileService;

    @Mock
    private HttpServletResponse response;

    @Mock
    private ServletOutputStream servletOutputStream;

    @Mock
    private ExportHandler exportHandler;

    @Mock
    private ImportHandler importHandler;

    private ImportExportService importExportService;

    @BeforeEach
    void setUp() throws IOException {
        importExportService = new ImportExportService(
                exportHandlerRegistry, importHandlerRegistry, fileGenerator, taskService, fileService);

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
        when(response.getOutputStream()).thenReturn(servletOutputStream);
    }

    // ==================== 导出：同步/异步判断 ====================

    @Test
    @DisplayName("导出 100 条(同步) - 直接返回 null,文件流写入响应")
    void testExport_Sync_ReturnsNull() {
        when(exportHandler.estimateCount(anyMap())).thenReturn(100L);
        ExportDataProvider provider = (pageNum, pageSize) -> List.of();
        when(exportHandler.getDataProvider(any())).thenReturn(provider);

        Object result = importExportService.export("user", Map.of(), "excel", null, null, response);

        assertNull(result, "同步导出应返回 null");
        verify(taskService, never()).createTask(any(), any());
        verify(response).setContentType("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
    }

    @Test
    @DisplayName("导出 50000 条(异步) - 返回 ExportTaskVO")
    void testExport_Async_ReturnsTaskVO() {
        when(exportHandler.estimateCount(anyMap())).thenReturn(50000L);
        TaskVO taskVO = new TaskVO();
        taskVO.setTaskId("task-async-001");
        taskVO.setStatus(TaskConstants.STATUS_PENDING);
        when(taskService.createTask(any(ExportTaskCreateForm.class), isNull())).thenReturn(taskVO);

        Object result = importExportService.export("user", Map.of(), "excel", null, null, response);

        assertInstanceOf(ExportTaskVO.class, result);
        ExportTaskVO vo = (ExportTaskVO) result;
        assertEquals("task-async-001", vo.getTaskId());
        assertEquals(TaskConstants.STATUS_PENDING, vo.getStatus());
        assertEquals(50000L, vo.getEstimatedCount());
        verify(fileService, never()).uploadFile(anyString(), any(), anyLong(), anyString());
    }

    @Test
    @DisplayName("导出强制同步(async=false) - 即使超过阈值也走同步")
    void testExport_ForceSync() {
        when(exportHandler.estimateCount(anyMap())).thenReturn(5000L);
        ExportDataProvider provider = (pageNum, pageSize) -> List.of();
        when(exportHandler.getDataProvider(any())).thenReturn(provider);

        Object result = importExportService.export("user", Map.of(), "excel", false, null, response);

        assertNull(result);
        verify(taskService, never()).createTask(any(), any());
    }

    @Test
    @DisplayName("导出强制异步(async=true) - 即使小于阈值也走异步")
    void testExport_ForceAsync() {
        when(exportHandler.estimateCount(anyMap())).thenReturn(10L);
        TaskVO taskVO = new TaskVO();
        taskVO.setTaskId("task-force-async");
        taskVO.setStatus(TaskConstants.STATUS_PENDING);
        when(taskService.createTask(any(ExportTaskCreateForm.class), isNull())).thenReturn(taskVO);

        Object result = importExportService.export("user", Map.of(), "excel", true, null, response);

        assertInstanceOf(ExportTaskVO.class, result);
        verify(fileService, never()).uploadFile(anyString(), any(), anyLong(), anyString());
    }

    @Test
    @DisplayName("导出超过 10 万条 - 抛 A0709 错误")
    void testExport_RowsExceedLimit() {
        when(exportHandler.estimateCount(anyMap())).thenReturn((long) (TaskConstants.MAX_ROWS + 1));

        BusinessException ex = assertThrows(BusinessException.class,
                () -> importExportService.export("user", Map.of(), "excel", null, null, response));
        assertEquals(ResultCode.EXPORT_ROWS_EXCEED_LIMIT.getCode(), ex.getResultCode().getCode());
    }

    @Test
    @DisplayName("导出 CSV 格式 - 设置 text/csv Content-Type")
    void testExport_CsvFormat() {
        when(exportHandler.estimateCount(anyMap())).thenReturn(10L);
        ExportDataProvider provider = (pageNum, pageSize) -> List.of();
        when(exportHandler.getDataProvider(any())).thenReturn(provider);

        importExportService.export("user", Map.of(), "csv", null, null, response);

        verify(response).setContentType("text/csv");
    }

    @Test
    @DisplayName("导出格式为空时默认 excel")
    void testExport_NullFormatDefaultsExcel() {
        when(exportHandler.estimateCount(anyMap())).thenReturn(10L);
        ExportDataProvider provider = (pageNum, pageSize) -> List.of();
        when(exportHandler.getDataProvider(any())).thenReturn(provider);

        importExportService.export("user", Map.of(), null, null, null, response);

        verify(response).setContentType("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
    }

    @Test
    @DisplayName("导出选中字段 - 通过 filterFields 过滤")
    void testExport_SelectedFields() {
        when(exportHandler.estimateCount(anyMap())).thenReturn(10L);
        ExportDataProvider provider = (pageNum, pageSize) -> List.of();
        when(exportHandler.getDataProvider(any())).thenReturn(provider);
        when(exportHandler.getFieldConfigs()).thenReturn(List.of(
                ExportFieldConfig.of("username", "用户名", 1),
                ExportFieldConfig.of("nickname", "昵称", 2)
        ));

        importExportService.export("user", Map.of(), "excel", null, List.of("username"), response);

        verify(exportHandler).getDataProvider(argThat(ctx ->
                ctx != null && ctx.getSelectedFields() != null
                        && ctx.getSelectedFields().contains("username")));
    }

    @Test
    @DisplayName("导出 handler 不存在 - 抛 A0710")
    void testExport_ModuleNotSupported() {
        when(exportHandlerRegistry.getHandler("unknown"))
                .thenThrow(new BusinessException(ResultCode.MODULE_IMPORT_NOT_SUPPORTED, "模块 unknown 不支持导出"));

        BusinessException ex = assertThrows(BusinessException.class,
                () -> importExportService.export("unknown", Map.of(), "excel", null, null, response));
        assertEquals(ResultCode.MODULE_IMPORT_NOT_SUPPORTED.getCode(), ex.getResultCode().getCode());
    }

    @Test
    @DisplayName("直接导出模式(useDirectExport=true) - 调用 handler.export 而非 fileGenerator")
    void testExport_DirectExport() throws Exception {
        when(exportHandler.estimateCount(anyMap())).thenReturn(10L);
        when(exportHandler.useDirectExport()).thenReturn(true);

        importExportService.export("user", Map.of(), "excel", null, null, response);

        verify(exportHandler, atLeastOnce()).export(any(ExportContext.class), any(ProgressCallback.class));
        verify(fileGenerator, never()).writeExcel(any(), anyList(), any());
    }

    // ==================== 导入：文件验证 ====================

    @Test
    @DisplayName("导入非 Excel/CSV 文件 - 返回 A0701 错误")
    void testImportData_UnsupportedFileType() {
        MockMultipartFile file = new MockMultipartFile(
                "file", "test.txt", "text/plain", "hello".getBytes(StandardCharsets.UTF_8));

        BusinessException ex = assertThrows(BusinessException.class,
                () -> importExportService.importData("user", file, "all", null, null));
        assertEquals(ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH.getCode(), ex.getResultCode().getCode());
    }

    @Test
    @DisplayName("导入 25MB 文件 - 返回 A0702 错误")
    void testImportData_FileSizeExceeds() {
        byte[] content = new byte[(int) (TaskConstants.MAX_IMPORT_FILE_SIZE + 1)];
        MockMultipartFile file = new MockMultipartFile(
                "file", "test.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", content);

        BusinessException ex = assertThrows(BusinessException.class,
                () -> importExportService.importData("user", file, "all", null, null));
        assertEquals(ResultCode.USER_UPLOAD_FILE_SIZE_EXCEEDS.getCode(), ex.getResultCode().getCode());
    }

    @Test
    @DisplayName("导入空文件 - 返回 A0703 错误")
    void testImportData_EmptyFile() {
        MockMultipartFile file = new MockMultipartFile(
                "file", "empty.xlsx", "application/octet-stream", new byte[0]);

        BusinessException ex = assertThrows(BusinessException.class,
                () -> importExportService.importData("user", file, "all", null, null));
        assertEquals(ResultCode.IMPORT_FILE_EMPTY.getCode(), ex.getResultCode().getCode());
    }

    @Test
    @DisplayName("导入文件名为空 - 返回 A0701 错误")
    void testImportData_NullFileName() {
        MockMultipartFile file = new MockMultipartFile(
                "file", null, "application/octet-stream", new byte[]{1, 2, 3});

        BusinessException ex = assertThrows(BusinessException.class,
                () -> importExportService.importData("user", file, "all", null, null));
        assertEquals(ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH.getCode(), ex.getResultCode().getCode());
    }

    // ==================== 导入：同步/异步判断 ====================

    @Test
    @DisplayName("导入 100 行(同步) - 返回 ImportResultVO")
    void testImportData_Sync() throws IOException {
        MultipartFile file = mockCsvFile(100);
        mockFileGeneratorParse(100);
        ImportResult result = ImportResult.success(100, 100);
        when(importHandler.importBatch(anyList(), any(ImportOptions.class), any())).thenReturn(result);

        Object ret = importExportService.importData("user", file, "all", null, null);

        assertInstanceOf(ImportResultVO.class, ret);
        ImportResultVO vo = (ImportResultVO) ret;
        assertEquals(100, vo.getTotalRows());
        assertEquals(100, vo.getSuccessCount());
        assertEquals(0, vo.getFailureCount());
        verify(taskService, never()).createTask(any(), any());
    }

    @Test
    @DisplayName("导入 50000 行(异步) - 返回 ImportTaskVO")
    void testImportData_Async() throws IOException {
        MultipartFile file = mockCsvFile(50000);
        mockFileGeneratorParse(50000);
        when(fileService.uploadFile(anyString(), any(InputStream.class), anyLong(), anyString()))
                .thenReturn("http://minio/temp/imports/abc.csv");
        TaskVO taskVO = new TaskVO();
        taskVO.setTaskId("task-import-001");
        taskVO.setStatus(TaskConstants.STATUS_PENDING);
        when(taskService.createTask(any(ExportTaskCreateForm.class), isNull())).thenReturn(taskVO);

        Object ret = importExportService.importData("user", file, "all", null, null);

        assertInstanceOf(ImportTaskVO.class, ret);
        ImportTaskVO vo = (ImportTaskVO) ret;
        assertEquals("task-import-001", vo.getTaskId());
        assertEquals(TaskConstants.STATUS_PENDING, vo.getStatus());
        verify(importHandler, never()).importBatch(anyList(), any(), any());
    }

    @Test
    @DisplayName("导入强制同步(async=false) - 即使超过阈值也走同步")
    void testImportData_ForceSync() throws IOException {
        MultipartFile file = mockCsvFile(5000);
        mockFileGeneratorParse(5000);
        ImportResult result = ImportResult.success(5000, 5000);
        when(importHandler.importBatch(anyList(), any(ImportOptions.class), any())).thenReturn(result);

        Object ret = importExportService.importData("user", file, "all", false, null);

        assertInstanceOf(ImportResultVO.class, ret);
        verify(taskService, never()).createTask(any(), any());
    }

    @Test
    @DisplayName("导入超过 10 万行 - 返回 A0708 错误")
    void testImportData_RowsExceedLimit() throws IOException {
        MultipartFile file = mockCsvFile(TaskConstants.MAX_ROWS + 1);
        mockFileGeneratorParse(TaskConstants.MAX_ROWS + 1);

        BusinessException ex = assertThrows(BusinessException.class,
                () -> importExportService.importData("user", file, "all", null, null));
        assertEquals(ResultCode.IMPORT_ROWS_EXCEED_LIMIT.getCode(), ex.getResultCode().getCode());
    }

    @Test
    @DisplayName("导入 handler 不存在 - 抛 A0710")
    void testImportData_ModuleNotSupported() {
        when(importHandlerRegistry.getHandler("unknown"))
                .thenThrow(new BusinessException(ResultCode.MODULE_IMPORT_NOT_SUPPORTED, "模块 unknown 不支持导入"));
        MockMultipartFile file = new MockMultipartFile(
                "file", "test.csv", "text/csv", "a,b\n1,2".getBytes(StandardCharsets.UTF_8));

        BusinessException ex = assertThrows(BusinessException.class,
                () -> importExportService.importData("unknown", file, "all", null, null));
        assertEquals(ResultCode.MODULE_IMPORT_NOT_SUPPORTED.getCode(), ex.getResultCode().getCode());
    }

    // ==================== 工具方法 ====================

    private MultipartFile mockCsvFile(int rows) throws IOException {
        StringBuilder sb = new StringBuilder("username\n");
        for (int i = 0; i < rows; i++) {
            sb.append("user").append(i).append("\n");
        }
        byte[] content = sb.toString().getBytes(StandardCharsets.UTF_8);
        return new MockMultipartFile("file", "test.csv", "text/csv", content);
    }

    private void mockFileGeneratorParse(int rowCount) throws IOException {
        try {
            doAnswer(invocation -> {
                ImportExportFileGenerator.RowConsumer consumer = invocation.getArgument(3);
                for (int i = 1; i <= rowCount; i++) {
                    consumer.consume(i, Map.of("username", "user" + i));
                }
                return null;
            }).when(fileGenerator).parse(any(InputStream.class), anyString(), anyList(), any());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
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

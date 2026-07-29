package com.pei.dehaze.service.importexport;

import cn.hutool.core.util.IdUtil;
import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.model.form.ExportTaskCreateForm;
import com.pei.dehaze.model.vo.ExportTaskVO;
import com.pei.dehaze.model.vo.ImportResultVO;
import com.pei.dehaze.model.vo.ImportTaskVO;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.TaskService;
import com.pei.dehaze.service.importexport.model.ExportContext;
import com.pei.dehaze.service.importexport.model.ExportFieldConfig;
import com.pei.dehaze.service.importexport.model.ImportFieldConfig;
import com.pei.dehaze.service.importexport.model.ImportOptions;
import com.pei.dehaze.service.importexport.model.ImportResult;
import com.pei.dehaze.service.strategy.ProgressCallback;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 通用导入导出服务
 * <p>负责同步导出/导入及异步任务创建。异步任务的实际执行由
 * GenericExportStrategy / GenericImportStrategy 承担，返回 TaskResult 后由
 * TaskExecutor 统一更新任务状态。
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ImportExportService {

    private final ExportHandlerRegistry exportHandlerRegistry;
    private final ImportHandlerRegistry importHandlerRegistry;
    private final ImportExportFileGenerator fileGenerator;
    private final TaskService taskService;
    private final FileService fileService;

    public Object export(String module, Map<String, Object> params, String format,
                         Boolean async, List<String> fields, HttpServletResponse response) {
        String actualFormat = (format == null || format.isBlank()) ? "excel" : format.toLowerCase();
        ExportHandler handler = exportHandlerRegistry.getHandler(module);

        long count = handler.estimateCount(params);
        if (count > TaskConstants.MAX_ROWS) {
            throw new BusinessException(ResultCode.EXPORT_ROWS_EXCEED_LIMIT,
                    "导出行数 " + count + " 超出限制 " + TaskConstants.MAX_ROWS);
        }

        boolean shouldAsync = async != null ? async : count > TaskConstants.SYNC_THRESHOLD || handler.useDirectExport();

        if (shouldAsync) {
            return createExportTask(module, params, actualFormat, fields, count);
        }

        writeSyncExport(handler, params, actualFormat, fields, response);
        return null;
    }

    private void writeSyncExport(ExportHandler handler, Map<String, Object> params, String format,
                                 List<String> fields, HttpServletResponse response) {
        boolean direct = handler.useDirectExport();
        String fileExt = direct ? "zip" : getFileExtension(format);
        String contentType = direct ? "application/zip" : getContentType(format);
        String fileName = handler.getModule() + "_export_" + System.currentTimeMillis() + "." + fileExt;

        response.setContentType(contentType);
        response.setHeader("Content-Disposition",
                "attachment; filename=" + URLEncoder.encode(fileName, StandardCharsets.UTF_8));

        try (OutputStream os = response.getOutputStream()) {
            ExportContext ctx = new ExportContext();
            ctx.setModule(handler.getModule());
            ctx.setFormat(format);
            ctx.setSelectedFields(fields);
            ctx.setQueryParams(params);
            ctx.setOutputStream(os);
            ctx.setTotalCount(handler.estimateCount(params));
            ctx.setAsync(false);

            if (direct) {
                handler.export(ctx, new NoopProgressCallback());
            } else {
                List<ExportFieldConfig> fieldConfigs = filterFields(handler.getFieldConfigs(), fields);
                if ("csv".equalsIgnoreCase(format)) {
                    fileGenerator.writeCsv(os, fieldConfigs, handler.getDataProvider(ctx));
                } else {
                    fileGenerator.writeExcel(os, fieldConfigs, handler.getDataProvider(ctx));
                }
            }
        } catch (Exception e) {
            response.reset();
            if (e instanceof BusinessException be) {
                throw be;
            }
            throw new BusinessException(ResultCode.SYSTEM_EXECUTION_ERROR, "导出失败: " + e.getMessage());
        }
    }

    public Object importData(String module, MultipartFile file, String mode,
                             Boolean async, Map<String, Object> extraParams) {
        validateUploadFile(file);

        ImportHandler handler = importHandlerRegistry.getHandler(module);

        long rowCount = countRows(file, handler.getDynamicFieldConfigs());
        if (rowCount > TaskConstants.MAX_ROWS) {
            throw new BusinessException(ResultCode.IMPORT_ROWS_EXCEED_LIMIT,
                    "导入行数 " + rowCount + " 超出限制 " + TaskConstants.MAX_ROWS);
        }

        boolean shouldAsync = async != null ? async : rowCount > TaskConstants.SYNC_THRESHOLD;

        if (shouldAsync) {
            return createImportTask(module, file, mode, extraParams);
        }

        return executeSyncImport(handler, file, mode, extraParams);
    }

    private ExportTaskVO createExportTask(String module, Map<String, Object> params, String format,
                                          List<String> fields, long count) {
        Map<String, Object> taskParams = new HashMap<>();
        taskParams.put("module", module);
        taskParams.put("format", format);
        taskParams.put("fields", fields);
        taskParams.put("query", params);

        ExportTaskCreateForm form = new ExportTaskCreateForm();
        form.setType(module + "_export");
        form.setParamsJson(JSONUtil.toJsonStr(taskParams));

        var taskVO = taskService.createTask(form, null);
        return ExportTaskVO.builder()
                .taskId(taskVO.getTaskId())
                .status(taskVO.getStatus())
                .estimatedCount(count)
                .build();
    }

    private ImportTaskVO createImportTask(String module, MultipartFile file, String mode,
                                          Map<String, Object> extraParams) {
        String objectName = "temp/imports/" + IdUtil.simpleUUID() + "/" + file.getOriginalFilename();
        try (InputStream is = file.getInputStream()) {
            String url = fileService.uploadFile(objectName, is, file.getSize(), file.getContentType());
            log.info("导入文件已上传: objectName={}, url={}", objectName, url);
        } catch (IOException e) {
            throw new BusinessException(ResultCode.USER_UPLOAD_FILE_ERROR, "文件上传失败: " + e.getMessage());
        }

        Map<String, Object> taskParams = new HashMap<>();
        taskParams.put("module", module);
        taskParams.put("fileObjectName", objectName);
        taskParams.put("mode", mode == null ? "all" : mode);
        taskParams.put("extraParams", extraParams);

        ExportTaskCreateForm form = new ExportTaskCreateForm();
        form.setType(module + "_import");
        form.setParamsJson(JSONUtil.toJsonStr(taskParams));

        var taskVO = taskService.createTask(form, null);
        return ImportTaskVO.builder()
                .taskId(taskVO.getTaskId())
                .status(taskVO.getStatus())
                .build();
    }

    private ImportResultVO executeSyncImport(ImportHandler handler, MultipartFile file,
                                             String mode, Map<String, Object> extraParams) {
        try {
            List<Map<String, Object>> rows = new ArrayList<>();
            fileGenerator.parse(file.getInputStream(), file.getOriginalFilename(),
                    handler.getDynamicFieldConfigs(), (rowNum, row) -> rows.add(row));

            ImportOptions options = ImportOptions.of(mode, extraParams);
            ImportResult result = handler.importBatch(rows, options, new NoopProgressCallback());

            String errorReportUrl = null;
            if (result.getFailureCount() > 0 && result.getErrors() != null && !result.getErrors().isEmpty()) {
                errorReportUrl = generateErrorReport(result.getErrors());
            }

            return ImportResultVO.from(result, errorReportUrl);
        } catch (IOException e) {
            throw new BusinessException(ResultCode.IMPORT_FILE_PARSE_ERROR, "文件解析失败: " + e.getMessage());
        }
    }

    private void validateUploadFile(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new BusinessException(ResultCode.IMPORT_FILE_EMPTY, "上传文件为空");
        }
        if (file.getSize() > TaskConstants.MAX_IMPORT_FILE_SIZE) {
            throw new BusinessException(ResultCode.USER_UPLOAD_FILE_SIZE_EXCEEDS,
                    "文件大小 " + file.getSize() + " 超出限制 " + TaskConstants.MAX_IMPORT_FILE_SIZE);
        }
        String name = file.getOriginalFilename();
        if (name == null) {
            throw new BusinessException(ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH, "文件名为空");
        }
        String lower = name.toLowerCase();
        if (!lower.endsWith(".xlsx") && !lower.endsWith(".xls") && !lower.endsWith(".csv")) {
            throw new BusinessException(ResultCode.USER_UPLOAD_FILE_TYPE_NOT_MATCH,
                    "不支持的文件类型: " + name);
        }
    }

    private long countRows(MultipartFile file, List<ImportFieldConfig> fields) {
        try {
            final int[] count = {0};
            fileGenerator.parse(file.getInputStream(), file.getOriginalFilename(), fields,
                    (rowNum, row) -> count[0]++);
            return count[0];
        } catch (IOException e) {
            throw new BusinessException(ResultCode.IMPORT_FILE_PARSE_ERROR, "文件解析失败: " + e.getMessage());
        }
    }

    private List<ExportFieldConfig> filterFields(List<ExportFieldConfig> all, List<String> selected) {
        if (selected == null || selected.isEmpty()) {
            return all.stream().filter(f -> !f.isHidden()).toList();
        }
        return all.stream()
                .filter(f -> selected.contains(f.getField()))
                .filter(f -> !f.isHidden())
                .toList();
    }

    private String generateErrorReport(List<ImportResult.ImportError> errors) {
        String objectName = "imports/" + IdUtil.simpleUUID() + "_errors.xlsx";
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            List<Map<String, Object>> errorRows = errors.stream()
                    .map(e -> {
                        Map<String, Object> row = new HashMap<>();
                        row.put("row", e.getRow());
                        row.put("field", e.getField() == null ? "" : e.getField());
                        row.put("message", e.getMessage());
                        return row;
                    })
                    .toList();

            List<ImportFieldConfig> errorFields = List.of(
                    ImportFieldConfig.of("row", "行号", true),
                    ImportFieldConfig.of("field", "字段", false),
                    ImportFieldConfig.of("message", "错误信息", true)
            );
            fileGenerator.writeTemplateExcel(baos, errorFields, errorRows);

            try (ByteArrayInputStream bis = new ByteArrayInputStream(baos.toByteArray())) {
                return fileService.uploadFile(objectName, bis, (long) baos.size(), getContentType("excel"));
            }
        } catch (IOException e) {
            log.warn("生成错误报告失败", e);
            return null;
        }
    }

    private String getFileExtension(String format) {
        return "csv".equalsIgnoreCase(format) ? "csv" : "xlsx";
    }

    private String getContentType(String format) {
        return "csv".equalsIgnoreCase(format)
                ? "text/csv"
                : "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
    }

    private static class NoopProgressCallback implements ProgressCallback {
        @Override
        public void updateProgress(int current, int total, String message) {
        }

        @Override
        public boolean isCancelled() {
            return false;
        }
    }
}

package com.pei.dehaze.service.strategy.impl;

import cn.hutool.json.JSONUtil;
import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.importexport.ImportExportFileGenerator;
import com.pei.dehaze.service.importexport.ImportHandler;
import com.pei.dehaze.service.importexport.ImportHandlerRegistry;
import com.pei.dehaze.service.importexport.model.ImportOptions;
import com.pei.dehaze.service.importexport.model.ImportResult;
import com.pei.dehaze.service.strategy.ProgressCallback;
import com.pei.dehaze.service.strategy.TaskResult;
import com.pei.dehaze.service.strategy.TaskStrategy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 通用导入策略
 * <p>处理所有 *_import 类型的任务，直接调用 {@link ImportHandler} 完成导入，
 * 返回 {@link TaskResult} 携带结果 JSON，由 {@link com.pei.dehaze.service.TaskExecutor} 统一更新任务状态。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class GenericImportStrategy implements TaskStrategy {

    private final ImportHandlerRegistry importHandlerRegistry;
    private final ImportExportFileGenerator fileGenerator;
    private final FileService fileService;

    @Override
    public List<String> getTaskTypes() {
        return List.of(
                TaskConstants.TYPE_USER_IMPORT,
                TaskConstants.TYPE_ROLE_IMPORT,
                TaskConstants.TYPE_DEPT_IMPORT,
                TaskConstants.TYPE_MENU_IMPORT,
                TaskConstants.TYPE_DICT_IMPORT,
                TaskConstants.TYPE_ALGORITHM_IMPORT
        );
    }

    @Override
    public TaskResult execute(SysTask task, Map<String, Object> params, ProgressCallback callback) {
        // 通用端点创建任务时 params 可能不含 module，从任务类型推导（如 user_import -> user）
        String module = (String) params.get("module");
        if (module == null || module.isBlank()) {
            module = TaskConstants.getModuleByType(task.getTaskType());
        }
        String fileObjectName = (String) params.get("fileObjectName");
        String mode = (String) params.getOrDefault("mode", "all");
        @SuppressWarnings("unchecked")
        Map<String, Object> extraParams = (Map<String, Object>) params.get("extraParams");

        ImportHandler handler = importHandlerRegistry.getHandler(module);

        try (InputStream is = fileService.downLoadFile(fileObjectName)) {
            List<Map<String, Object>> rows = new ArrayList<>();
            fileGenerator.parse(is, fileObjectName, handler.getDynamicFieldConfigs(),
                    (rowNum, row) -> rows.add(row));

            ImportOptions options = ImportOptions.of(mode, extraParams);
            ImportResult result = handler.importBatch(rows, options, callback);

            String errorReportUrl = null;
            if (result.getFailureCount() > 0 && result.getErrors() != null && !result.getErrors().isEmpty()) {
                errorReportUrl = generateErrorReport(task.getTaskId(), result.getErrors());
            }

            String finalResult = errorReportUrl != null
                    ? buildResultJson(result, errorReportUrl)
                    : JSONUtil.toJsonStr(result);

            log.info("异步导入完成: taskId={}, module={}, success={}, failure={}",
                    task.getTaskId(), module, result.getSuccessCount(), result.getFailureCount());
            return TaskResult.success(finalResult);
        } catch (Exception e) {
            log.error("异步导入失败: taskId={}, module={}", task.getTaskId(), module, e);
            return TaskResult.failure("导入失败: " + e.getMessage());
        }
    }

    private String generateErrorReport(String taskId, List<ImportResult.ImportError> errors) {
        String objectName = "imports/" + taskId + "_errors.xlsx";
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

            List<com.pei.dehaze.service.importexport.model.ImportFieldConfig> errorFields = List.of(
                    com.pei.dehaze.service.importexport.model.ImportFieldConfig.of("row", "行号", true),
                    com.pei.dehaze.service.importexport.model.ImportFieldConfig.of("field", "字段", false),
                    com.pei.dehaze.service.importexport.model.ImportFieldConfig.of("message", "错误信息", true)
            );
            fileGenerator.writeTemplateExcel(baos, errorFields, errorRows);

            try (ByteArrayInputStream bis = new ByteArrayInputStream(baos.toByteArray())) {
                return fileService.uploadFile(objectName, bis, (long) baos.size(), getContentType("excel"));
            }
        } catch (Exception e) {
            log.warn("生成错误报告失败: taskId={}", taskId, e);
            return null;
        }
    }

    private String buildResultJson(ImportResult result, String errorReportUrl) {
        Map<String, Object> map = new HashMap<>();
        map.put("totalRows", result.getTotalRows());
        map.put("successCount", result.getSuccessCount());
        map.put("failureCount", result.getFailureCount());
        map.put("skippedCount", result.getSkippedCount());
        map.put("errors", result.getErrors());
        map.put("errorReportUrl", errorReportUrl);
        return JSONUtil.toJsonStr(map);
    }

    private String getContentType(String format) {
        return "csv".equalsIgnoreCase(format)
                ? "text/csv"
                : "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
    }
}

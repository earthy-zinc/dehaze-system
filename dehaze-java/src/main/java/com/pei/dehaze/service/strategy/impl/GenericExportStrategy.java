package com.pei.dehaze.service.strategy.impl;

import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.service.importexport.ExportHandler;
import com.pei.dehaze.service.importexport.ExportHandlerRegistry;
import com.pei.dehaze.service.importexport.ImportExportFileGenerator;
import com.pei.dehaze.service.importexport.model.ExportContext;
import com.pei.dehaze.service.importexport.model.ExportDataProvider;
import com.pei.dehaze.service.importexport.model.ExportFieldConfig;
import com.pei.dehaze.service.strategy.ProgressCallback;
import com.pei.dehaze.service.strategy.TaskResult;
import com.pei.dehaze.service.strategy.TaskStrategy;
import com.pei.dehaze.service.impl.file.StorageServiceFactory;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 通用导出策略
 * <p>处理所有 *_export 类型的任务，直接调用 {@link ExportHandler} 完成导出，
 * 返回 {@link TaskResult} 携带下载链接，由 {@link com.pei.dehaze.service.TaskExecutor} 统一更新任务状态。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class GenericExportStrategy implements TaskStrategy {

    private final ExportHandlerRegistry exportHandlerRegistry;
    private final ImportExportFileGenerator fileGenerator;
    private final StorageServiceFactory storageServiceFactory;

    @Override
    public List<String> getTaskTypes() {
        return List.of(
                TaskConstants.TYPE_USER_EXPORT,
                TaskConstants.TYPE_ROLE_EXPORT,
                TaskConstants.TYPE_DEPT_EXPORT,
                TaskConstants.TYPE_MENU_EXPORT,
                TaskConstants.TYPE_DICT_EXPORT,
                TaskConstants.TYPE_DATASET_EXPORT,
                TaskConstants.TYPE_ALGORITHM_EXPORT
        );
    }

    @Override
    public TaskResult execute(SysTask task, Map<String, Object> params, ProgressCallback callback) {
        // 通用端点创建任务时 params 可能不含 module，从任务类型推导（如 user_export -> user）
        String module = (String) params.get("module");
        if (module == null || module.isBlank()) {
            module = TaskConstants.getModuleByType(task.getTaskType());
        }
        String format = (String) params.getOrDefault("format", "excel");
        @SuppressWarnings("unchecked")
        List<String> fields = (List<String>) params.get("fields");

        ExportHandler handler = exportHandlerRegistry.getHandler(module);

        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            ExportContext ctx = new ExportContext();
            ctx.setTaskId(task.getTaskId());
            ctx.setModule(module);
            ctx.setFormat(format);
            ctx.setSelectedFields(fields);
            ctx.setQueryParams(extractQueryParams(params));
            ctx.setOutputStream(baos);
            ctx.setTotalCount(handler.estimateCount(ctx.getQueryParams()));
            ctx.setAsync(true);

            callback.updateProgress(0, (int) Math.min(ctx.getTotalCount(), Integer.MAX_VALUE), "开始导出");

            if (handler.useDirectExport()) {
                handler.export(ctx, callback);
            } else {
                List<ExportFieldConfig> fieldConfigs = filterFields(handler.getFieldConfigs(), fields);
                ExportDataProvider baseProvider = handler.getDataProvider(ctx);
                ExportDataProvider wrappedProvider = (pageNum, pageSize) -> {
                    List<List<Object>> batch = baseProvider.fetchBatch(pageNum, pageSize);
                    if (!batch.isEmpty()) {
                        int processed = pageNum * pageSize;
                        int total = (int) Math.min(ctx.getTotalCount(), Integer.MAX_VALUE);
                        callback.updateProgress(Math.min(processed, total), total,
                                "导出中: " + Math.min(processed, (int) ctx.getTotalCount()) + "/" + ctx.getTotalCount());
                    }
                    return batch;
                };

                if ("csv".equalsIgnoreCase(format)) {
                    fileGenerator.writeCsv(baos, fieldConfigs, wrappedProvider);
                } else {
                    fileGenerator.writeExcel(baos, fieldConfigs, wrappedProvider);
                }
            }

            String fileExt = handler.useDirectExport() ? "zip" : getFileExtension(format);
            String contentType = handler.useDirectExport() ? "application/zip" : getContentType(format);
            String objectName = "exports/" + task.getTaskId() + "." + fileExt;
            // uploadFile 返回 objectName（落库 sys_task.result），下载时由 getDownloadUrl 动态拼接 URL
            String resultObjectName = storageServiceFactory.getDefault().uploadFile(objectName,
                    new ByteArrayInputStream(baos.toByteArray()),
                    (long) baos.size(),
                    contentType);

            log.debug("异步导出完成: taskId={}, module={}, objectName={}", task.getTaskId(), module, resultObjectName);
            return TaskResult.success(resultObjectName);
        } catch (Exception e) {
            log.error("异步导出失败: taskId={}, module={}", task.getTaskId(), module, e);
            return TaskResult.failure("导出失败: " + e.getMessage());
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

    @SuppressWarnings("unchecked")
    private Map<String, Object> extractQueryParams(Map<String, Object> params) {
        Object query = params.get("query");
        if (query instanceof Map<?, ?> queryMap) {
            return (Map<String, Object>) queryMap;
        }
        return new HashMap<>();
    }

    private String getFileExtension(String format) {
        return "csv".equalsIgnoreCase(format) ? "csv" : "xlsx";
    }

    private String getContentType(String format) {
        return "csv".equalsIgnoreCase(format)
                ? "text/csv"
                : "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
    }
}

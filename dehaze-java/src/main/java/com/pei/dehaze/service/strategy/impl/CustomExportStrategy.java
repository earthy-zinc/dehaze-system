package com.pei.dehaze.service.strategy.impl;

import cn.hutool.core.util.IdUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.model.form.ExportTaskCreateForm;
import com.pei.dehaze.service.SysDatasetItemService;
import com.pei.dehaze.service.SysItemFileService;
import com.pei.dehaze.service.strategy.AbstractExportStrategy;
import com.pei.dehaze.service.strategy.ProgressCallback;
import com.pei.dehaze.service.strategy.TaskResult;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.FileOutputStream;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipOutputStream;

/**
 * 自定义导出策略
 * 支持基于筛选条件的自定义导出
 */
@Slf4j
@Component
public class CustomExportStrategy extends AbstractExportStrategy {

    @Resource
    private SysDatasetItemService sysDatasetItemService;

    @Resource
    private SysItemFileService sysItemFileService;

    @Override
    public String getTaskType() {
        return TaskConstants.TYPE_CUSTOM_EXPORT;
    }

    @Override
    public void validateParams(Map<String, Object> params) {
        Object datasetId = params.get("targetId");
        Object targetIds = params.get("targetIds");
        Object filters = params.get("filters");
        
        if (datasetId == null && targetIds == null && filters == null) {
            throw new BusinessException("自定义导出需要指定数据集ID、数据项ID列表或筛选条件");
        }
    }

    @Override
    @SuppressWarnings("unchecked")
    public TaskResult execute(SysTask task, Map<String, Object> params, ProgressCallback callback) {
        log.info("开始执行自定义导出: taskId={}", task.getTaskId());

        ExportTaskCreateForm.ExportOptions options = getExportOptions(params);
        List<SysDatasetItem> items = resolveItems(params);
        
        if (items.isEmpty()) {
            return TaskResult.failure("未找到符合条件的数据项");
        }

        File zipFile = null;
        try {
            zipFile = createTempZipFile("custom_export_" + IdUtil.simpleUUID());
            String downloadUrl = exportItemsToZip(zipFile, items, options, callback);

            log.info("自定义导出完成: taskId={}, itemCount={}", task.getTaskId(), items.size());
            return TaskResult.success(downloadUrl, Map.of("itemCount", items.size()));

        } catch (Exception e) {
            log.error("自定义导出失败: taskId={}", task.getTaskId(), e);
            cleanupTempFile(zipFile);
            return TaskResult.failure("自定义导出失败: " + e.getMessage());
        }
    }

    @SuppressWarnings("unchecked")
    private List<SysDatasetItem> resolveItems(Map<String, Object> params) {
        Object targetIds = params.get("targetIds");
        if (targetIds instanceof List<?> list && !list.isEmpty()) {
            List<Long> itemIds = ((List<Number>) targetIds).stream()
                    .map(Number::longValue)
                    .toList();
            return sysDatasetItemService.listByIds(itemIds);
        }

        Object targetId = params.get("targetId");
        if (targetId != null) {
            Long datasetId = ((Number) targetId).longValue();
            LambdaQueryWrapper<SysDatasetItem> query = new LambdaQueryWrapper<SysDatasetItem>()
                    .eq(SysDatasetItem::getDatasetId, datasetId);
            
            Object filters = params.get("filters");
            if (filters instanceof Map<?, ?> filterMap) {
                applyFilters(query, (Map<String, Object>) filterMap);
            }
            
            return sysDatasetItemService.list(query);
        }

        return List.of();
    }

    private void applyFilters(LambdaQueryWrapper<SysDatasetItem> query, Map<String, Object> filters) {
        Object name = filters.get("name");
        if (name instanceof String nameStr && !nameStr.isBlank()) {
            query.like(SysDatasetItem::getName, nameStr);
        }
    }

    private String exportItemsToZip(File zipFile, List<SysDatasetItem> items,
                                    ExportTaskCreateForm.ExportOptions options,
                                    ProgressCallback callback) throws Exception {
        String structure = options.getStructure();
        List<String> includeTypes = options.getIncludeTypes();
        Boolean includeThumbnail = options.getIncludeThumbnail();

        int totalFiles = 0;
        for (SysDatasetItem item : items) {
            long fileCount = sysItemFileService.count(
                    new LambdaQueryWrapper<SysItemFile>()
                            .eq(SysItemFile::getItemId, item.getId())
            );
            totalFiles += (int) fileCount;
            if (Boolean.TRUE.equals(includeThumbnail)) {
                totalFiles += (int) fileCount;
            }
        }

        callback.updateProgress(0, totalFiles, "开始自定义导出");

        int processedFiles = 0;
        try (FileOutputStream fos = new FileOutputStream(zipFile);
             ZipOutputStream zos = new ZipOutputStream(fos)) {

            for (SysDatasetItem item : items) {
                callback.checkCancelled();

                List<SysItemFile> itemFiles = sysItemFileService.list(
                        new LambdaQueryWrapper<SysItemFile>()
                                .eq(SysItemFile::getItemId, item.getId())
                );

                for (SysItemFile itemFile : itemFiles) {
                    callback.checkCancelled();

                    if (shouldIncludeType(includeTypes, itemFile.getType())) {
                        addFileToZip(zos, itemFile, structure, item.getName(), null);
                        processedFiles++;
                        callback.updateProgress(processedFiles, totalFiles, "正在导出: " + item.getName());
                    }

                    if (Boolean.TRUE.equals(includeThumbnail)) {
                        addFileToZip(zos, itemFile, structure, item.getName(), "thumbnail");
                        processedFiles++;
                        callback.updateProgress(processedFiles, totalFiles, "导出缩略图");
                    }
                }
            }
        }

        return uploadZipFile(zipFile, zipFile.getName().replace(".zip", ""));
    }
}

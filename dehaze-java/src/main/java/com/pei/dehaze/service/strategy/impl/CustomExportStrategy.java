package com.pei.dehaze.service.strategy.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.model.form.ExportTaskCreateForm;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.SysDatasetItemService;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.SysItemFileService;
import com.pei.dehaze.service.strategy.AbstractExportStrategy;
import com.pei.dehaze.service.strategy.ProgressCallback;
import com.pei.dehaze.service.strategy.TaskResult;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.File;
import java.util.List;
import java.util.Map;

/**
 * 自定义导出策略
 * 支持基于筛选条件的自定义导出
 */
@Slf4j
@Component
public class CustomExportStrategy extends AbstractExportStrategy {

    private final SysDatasetItemService sysDatasetItemService;

    public CustomExportStrategy(SysFileService sysFileService, FileService fileService,
                                SysItemFileService sysItemFileService,
                                SysDatasetItemService sysDatasetItemService) {
        super(sysFileService, fileService, sysItemFileService);
        this.sysDatasetItemService = sysDatasetItemService;
    }

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
            zipFile = createTempZipFile("custom_export");
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
}

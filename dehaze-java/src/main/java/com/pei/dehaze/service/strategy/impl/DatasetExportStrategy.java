package com.pei.dehaze.service.strategy.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.model.form.ExportTaskCreateForm;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.SysDatasetItemService;
import com.pei.dehaze.service.SysDatasetService;
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
 * 数据集导出策略
 * 导出整个数据集的所有数据项
 */
@Slf4j
@Component
public class DatasetExportStrategy extends AbstractExportStrategy {

    private final SysDatasetService sysDatasetService;
    private final SysDatasetItemService sysDatasetItemService;

    public DatasetExportStrategy(SysFileService sysFileService, FileService fileService,
                                 SysItemFileService sysItemFileService,
                                 SysDatasetService sysDatasetService,
                                 SysDatasetItemService sysDatasetItemService) {
        super(sysFileService, fileService, sysItemFileService);
        this.sysDatasetService = sysDatasetService;
        this.sysDatasetItemService = sysDatasetItemService;
    }

    @Override
    public String getTaskType() {
        return TaskConstants.TYPE_DATASET_EXPORT;
    }

    @Override
    public void validateParams(Map<String, Object> params) {
        Object targetId = params.get("targetId");
        if (targetId == null) {
            throw new BusinessException("数据集ID不能为空");
        }
    }

    @Override
    public TaskResult execute(SysTask task, Map<String, Object> params, ProgressCallback callback) {
        Long datasetId = ((Number) params.get("targetId")).longValue();
        ExportTaskCreateForm.ExportOptions options = getExportOptions(params);

        log.info("开始执行数据集导出: taskId={}, datasetId={}", task.getTaskId(), datasetId);

        SysDataset dataset = sysDatasetService.getById(datasetId);
        if (dataset == null) {
            return TaskResult.failure("数据集不存在");
        }

        List<SysDatasetItem> items = sysDatasetItemService.list(
                new LambdaQueryWrapper<SysDatasetItem>()
                        .eq(SysDatasetItem::getDatasetId, datasetId)
        );

        if (items.isEmpty()) {
            return TaskResult.failure("数据集为空，无可导出内容");
        }

        File zipFile = null;
        try {
            zipFile = createTempZipFile(dataset.getName() + "_export");
            String downloadUrl = exportItemsToZip(zipFile, items, options, callback);

            log.info("数据集导出完成: taskId={}, itemCount={}", task.getTaskId(), items.size());
            return TaskResult.success(downloadUrl, Map.of(
                    "datasetId", datasetId,
                    "datasetName", dataset.getName(),
                    "itemCount", items.size()
            ));

        } catch (Exception e) {
            log.error("数据集导出失败: taskId={}", task.getTaskId(), e);
            cleanupTempFile(zipFile);
            return TaskResult.failure("导出失败: " + e.getMessage());
        }
    }
}

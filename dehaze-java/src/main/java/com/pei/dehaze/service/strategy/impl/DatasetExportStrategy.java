package com.pei.dehaze.service.strategy.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.entity.SysTask;
import com.pei.dehaze.model.form.ExportTaskCreateForm;
import com.pei.dehaze.service.SysDatasetItemService;
import com.pei.dehaze.service.SysDatasetService;
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
 * 数据集导出策略
 * 导出整个数据集的所有数据项
 */
@Slf4j
@Component
public class DatasetExportStrategy extends AbstractExportStrategy {

    @Resource
    private SysDatasetService sysDatasetService;

    @Resource
    private SysDatasetItemService sysDatasetItemService;

    @Resource
    private SysItemFileService sysItemFileService;

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

        // 验证数据集
        SysDataset dataset = sysDatasetService.getById(datasetId);
        if (dataset == null) {
            return TaskResult.failure("数据集不存在");
        }

        // 查询数据项
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

    @Override
    public void cancel(SysTask task) {
        log.info("取消数据集导出任务: taskId={}", task.getTaskId());
    }

    /**
     * 将数据项列表导出为ZIP
     */
    private String exportItemsToZip(File zipFile, List<SysDatasetItem> items,
                                    ExportTaskCreateForm.ExportOptions options,
                                    ProgressCallback callback) throws Exception {
        String structure = options.getStructure();
        List<String> includeTypes = options.getIncludeTypes();
        Boolean includeThumbnail = options.getIncludeThumbnail();

        // 计算总文件数
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

        callback.updateProgress(0, totalFiles, "开始导出");

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

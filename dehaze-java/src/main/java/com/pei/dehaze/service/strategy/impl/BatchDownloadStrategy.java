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
 * 批量下载策略
 * 批量下载多个数据项
 */
@Slf4j
@Component
public class BatchDownloadStrategy extends AbstractExportStrategy {

    @Resource
    private SysDatasetItemService sysDatasetItemService;

    @Resource
    private SysItemFileService sysItemFileService;

    @Override
    public String getTaskType() {
        return TaskConstants.TYPE_BATCH_DOWNLOAD;
    }

    @Override
    public void validateParams(Map<String, Object> params) {
        Object targetIds = params.get("targetIds");
        if (targetIds == null) {
            throw new BusinessException("数据项ID列表不能为空");
        }
        if (targetIds instanceof List<?> list && list.isEmpty()) {
            throw new BusinessException("数据项ID列表不能为空");
        }
    }

    @Override
    @SuppressWarnings("unchecked")
    public TaskResult execute(SysTask task, Map<String, Object> params, ProgressCallback callback) {
        List<Long> itemIds = ((List<Number>) params.get("targetIds"))
                .stream()
                .map(Number::longValue)
                .toList();
        ExportTaskCreateForm.ExportOptions options = getExportOptions(params);

        log.info("开始执行批量下载: taskId={}, itemCount={}", task.getTaskId(), itemIds.size());

        // 验证数据项存在性
        List<SysDatasetItem> items = sysDatasetItemService.listByIds(itemIds);
        if (items.isEmpty()) {
            return TaskResult.failure("未找到有效的数据项");
        }

        File zipFile = null;
        try {
            zipFile = createTempZipFile("batch_download_" + IdUtil.simpleUUID());
            String downloadUrl = exportItemsToZip(zipFile, items, options, callback);

            log.info("批量下载完成: taskId={}, itemCount={}", task.getTaskId(), items.size());
            return TaskResult.success(downloadUrl, Map.of(
                    "itemCount", items.size(),
                    "requestedCount", itemIds.size()
            ));

        } catch (Exception e) {
            log.error("批量下载失败: taskId={}", task.getTaskId(), e);
            cleanupTempFile(zipFile);
            return TaskResult.failure("批量下载失败: " + e.getMessage());
        }
    }

    /**
     * 将多个数据项导出为ZIP
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

        callback.updateProgress(0, totalFiles, "开始批量下载");

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
                        callback.updateProgress(processedFiles, totalFiles, "正在下载: " + item.getName());
                    }

                    if (Boolean.TRUE.equals(includeThumbnail)) {
                        addFileToZip(zos, itemFile, structure, item.getName(), "thumbnail");
                        processedFiles++;
                        callback.updateProgress(processedFiles, totalFiles, "下载缩略图");
                    }
                }
            }
        }

        return uploadZipFile(zipFile, zipFile.getName().replace(".zip", ""));
    }
}

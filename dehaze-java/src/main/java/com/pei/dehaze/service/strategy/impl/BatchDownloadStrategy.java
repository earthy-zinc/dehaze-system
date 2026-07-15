package com.pei.dehaze.service.strategy.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.constant.TaskConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysItemFile;
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
import java.io.FileOutputStream;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.zip.ZipOutputStream;

/**
 * 批量下载策略
 * 批量下载多个数据项
 */
@Slf4j
@Component
public class BatchDownloadStrategy extends AbstractExportStrategy {

    private final SysDatasetItemService sysDatasetItemService;

    public BatchDownloadStrategy(SysFileService sysFileService, FileService fileService,
                                 SysItemFileService sysItemFileService,
                                 SysDatasetItemService sysDatasetItemService) {
        super(sysFileService, fileService, sysItemFileService);
        this.sysDatasetItemService = sysDatasetItemService;
    }

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

        List<SysDatasetItem> items = sysDatasetItemService.listByIds(itemIds);
        if (items.isEmpty()) {
            return TaskResult.failure("未找到有效的数据项");
        }

        File zipFile = null;
        try {
            zipFile = createTempZipFile("batch_download");
            String downloadUrl = exportItemsToZipBatch(zipFile, items, options, callback);

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
     * 批量查询所有数据项的文件并导出为ZIP（避免逐项 count + list 的 N+1 查询）
     */
    private String exportItemsToZipBatch(File zipFile, List<SysDatasetItem> items,
                                         ExportTaskCreateForm.ExportOptions options,
                                         ProgressCallback callback) throws Exception {
        String structure = options.getStructure();
        List<String> includeTypes = options.getIncludeTypes();
        Boolean includeThumbnail = options.getIncludeThumbnail();

        List<Long> itemIds = items.stream().map(SysDatasetItem::getId).toList();
        Map<Long, List<SysItemFile>> itemFilesMap = sysItemFileService.list(
                        new LambdaQueryWrapper<SysItemFile>()
                                .in(SysItemFile::getItemId, itemIds))
                .stream()
                .collect(Collectors.groupingBy(SysItemFile::getItemId));

        int totalFiles = 0;
        for (SysDatasetItem item : items) {
            int fileCount = itemFilesMap.getOrDefault(item.getId(), Collections.emptyList()).size();
            totalFiles += fileCount;
            if (Boolean.TRUE.equals(includeThumbnail)) {
                totalFiles += fileCount;
            }
        }

        callback.updateProgress(0, totalFiles, "开始批量下载");

        int processedFiles = 0;
        try (FileOutputStream fos = new FileOutputStream(zipFile);
             ZipOutputStream zos = new ZipOutputStream(fos)) {

            for (SysDatasetItem item : items) {
                callback.checkCancelled();

                List<SysItemFile> itemFiles = itemFilesMap.getOrDefault(item.getId(), Collections.emptyList());

                for (SysItemFile itemFile : itemFiles) {
                    callback.checkCancelled();

                    if (shouldIncludeType(includeTypes, itemFile.getType())) {
                        addFileToZip(zos, itemFile, structure, item.getName(), null);
                        processedFiles++;
                        callback.updateProgress(processedFiles, totalFiles, "正在下载: " + item.getName());
                    }

                    if (Boolean.TRUE.equals(includeThumbnail)) {
                        addFileToZip(zos, itemFile, structure, item.getName(), THUMBNAIL_SUBFOLDER);
                        processedFiles++;
                        callback.updateProgress(processedFiles, totalFiles, "下载缩略图");
                    }
                }
            }
        }

        return uploadZipFile(zipFile, zipFile.getName().replace(".zip", ""));
    }
}

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
import java.util.List;
import java.util.Map;
import java.util.zip.ZipOutputStream;

/**
 * 数据项下载策略
 * 下载单个数据项的所有文件
 */
@Slf4j
@Component
public class ItemDownloadStrategy extends AbstractExportStrategy {

    private final SysDatasetItemService sysDatasetItemService;

    public ItemDownloadStrategy(SysFileService sysFileService, FileService fileService,
                                SysItemFileService sysItemFileService,
                                SysDatasetItemService sysDatasetItemService) {
        super(sysFileService, fileService, sysItemFileService);
        this.sysDatasetItemService = sysDatasetItemService;
    }

    @Override
    public String getTaskType() {
        return TaskConstants.TYPE_ITEM_DOWNLOAD;
    }

    @Override
    public void validateParams(Map<String, Object> params) {
        Object targetId = params.get("targetId");
        if (targetId == null) {
            throw new BusinessException("数据项ID不能为空");
        }
    }

    @Override
    public TaskResult execute(SysTask task, Map<String, Object> params, ProgressCallback callback) {
        Long itemId = ((Number) params.get("targetId")).longValue();
        ExportTaskCreateForm.ExportOptions options = getExportOptions(params);

        log.info("开始执行数据项下载: taskId={}, itemId={}", task.getTaskId(), itemId);

        SysDatasetItem item = sysDatasetItemService.getById(itemId);
        if (item == null) {
            return TaskResult.failure("数据项不存在");
        }

        List<SysItemFile> itemFiles = sysItemFileService.list(
                new LambdaQueryWrapper<SysItemFile>()
                        .eq(SysItemFile::getItemId, itemId)
        );

        if (itemFiles.isEmpty()) {
            return TaskResult.failure("数据项无文件可下载");
        }

        File zipFile = null;
        try {
            zipFile = createTempZipFile(item.getName() + "_download");
            String downloadUrl = exportFilesToZip(zipFile, item, itemFiles, options, callback);

            log.info("数据项下载完成: taskId={}, fileCount={}", task.getTaskId(), itemFiles.size());
            return TaskResult.success(downloadUrl, Map.of(
                    "itemId", itemId,
                    "itemName", item.getName(),
                    "fileCount", itemFiles.size()
            ));

        } catch (Exception e) {
            log.error("数据项下载失败: taskId={}", task.getTaskId(), e);
            cleanupTempFile(zipFile);
            return TaskResult.failure("下载失败: " + e.getMessage());
        }
    }

    /**
     * 将单个数据项的文件导出为ZIP
     */
    private String exportFilesToZip(File zipFile, SysDatasetItem item, List<SysItemFile> itemFiles,
                                    ExportTaskCreateForm.ExportOptions options,
                                    ProgressCallback callback) throws Exception {
        String structure = options.getStructure();
        List<String> includeTypes = options.getIncludeTypes();
        Boolean includeThumbnail = options.getIncludeThumbnail();

        int totalFiles = itemFiles.size();
        if (Boolean.TRUE.equals(includeThumbnail)) {
            totalFiles *= 2;
        }

        callback.updateProgress(0, totalFiles, "开始下载");

        int processedFiles = 0;
        try (FileOutputStream fos = new FileOutputStream(zipFile);
             ZipOutputStream zos = new ZipOutputStream(fos)) {

            for (SysItemFile itemFile : itemFiles) {
                callback.checkCancelled();

                if (shouldIncludeType(includeTypes, itemFile.getType())) {
                    addFileToZip(zos, itemFile, structure, item.getName(), null);
                    processedFiles++;
                    callback.updateProgress(processedFiles, totalFiles, "正在下载文件");
                }

                if (Boolean.TRUE.equals(includeThumbnail)) {
                    addFileToZip(zos, itemFile, structure, item.getName(), THUMBNAIL_SUBFOLDER);
                    processedFiles++;
                    callback.updateProgress(processedFiles, totalFiles, "下载缩略图");
                }
            }
        }

        return uploadZipFile(zipFile, zipFile.getName().replace(".zip", ""));
    }
}

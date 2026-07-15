package com.pei.dehaze.service.strategy;

import cn.hutool.core.util.IdUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.form.ExportTaskCreateForm;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.SysItemFileService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * 导出策略抽象基类
 * 提供ZIP打包、文件上传等公共能力
 */
@Slf4j
@RequiredArgsConstructor
public abstract class AbstractExportStrategy implements TaskStrategy {

    protected static final String TEMP_DIR = System.getProperty("java.io.tmpdir") + "export/";
    protected static final String EXPORT_PREFIX = "export_";

    private static final String STRUCTURE_BY_ITEM = "by_item";
    protected static final String THUMBNAIL_SUBFOLDER = "thumbnail";
    private static final String DEFAULT_FILE_EXTENSION = ".jpg";
    private static final String ZIP_CONTENT_TYPE = "application/zip";
    private static final String EXPORT_OBJECT_PREFIX = "exports/";
    private static final int ZIP_BUFFER_SIZE = 8192;

    protected final SysFileService sysFileService;
    protected final FileService fileService;
    protected final SysItemFileService sysItemFileService;

    /**
     * 从参数中获取导出选项
     */
    protected ExportTaskCreateForm.ExportOptions getExportOptions(Map<String, Object> params) {
        Object optionsObj = params.get("options");
        if (optionsObj instanceof ExportTaskCreateForm.ExportOptions options) {
            return options;
        }
        ExportTaskCreateForm.ExportOptions options = new ExportTaskCreateForm.ExportOptions();
        if (optionsObj instanceof Map<?, ?> optionsMap) {
            if (optionsMap.get("structure") instanceof String structure) {
                options.setStructure(structure);
            }
            if (optionsMap.get("includeTypes") instanceof List<?> list) {
                options.setIncludeTypes(list.stream()
                        .filter(String.class::isInstance)
                        .map(String.class::cast)
                        .toList());
            }
            if (optionsMap.get("includeThumbnail") instanceof Boolean includeThumbnail) {
                options.setIncludeThumbnail(includeThumbnail);
            }
        }
        return options;
    }

    /**
     * 判断是否应该包含该文件类型
     */
    protected boolean shouldIncludeType(List<String> includeTypes, String type) {
        if (includeTypes == null || includeTypes.isEmpty()) {
            return true;
        }
        return includeTypes.contains(type);
    }

    /**
     * 将数据项列表导出为ZIP
     */
    protected String exportItemsToZip(File zipFile, List<SysDatasetItem> items,
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
                        addFileToZip(zos, itemFile, structure, item.getName(), THUMBNAIL_SUBFOLDER);
                        processedFiles++;
                        callback.updateProgress(processedFiles, totalFiles, "导出缩略图");
                    }
                }
            }
        }

        return uploadZipFile(zipFile, zipFile.getName().replace(".zip", ""));
    }

    /**
     * 添加文件到ZIP
     */
    protected void addFileToZip(ZipOutputStream zos, SysItemFile itemFile,
                                String structure, String itemName, String subfolder) throws Exception {
        SysFile sysFile = sysFileService.getById(itemFile.getFileId());
        if (sysFile == null || sysFile.getObjectName() == null) {
            log.warn("文件不存在或objectName为空: fileId={}", itemFile.getFileId());
            return;
        }

        String entryPath = buildZipEntryPath(structure, itemName, subfolder, itemFile.getId(), sysFile.getName());
        ZipEntry zipEntry = new ZipEntry(entryPath);
        zos.putNextEntry(zipEntry);

        try (InputStream inputStream = fileService.downLoadFile(sysFile.getObjectName())) {
            byte[] buffer = new byte[ZIP_BUFFER_SIZE];
            int len;
            while ((len = inputStream.read(buffer)) > 0) {
                zos.write(buffer, 0, len);
            }
        }
        zos.closeEntry();
    }

    /**
     * 构建ZIP条目路径
     */
    private String buildZipEntryPath(String structure, String itemName, String subfolder, Long fileId, String fileName) {
        String extension = getFileExtension(fileName);
        String baseName = fileId + extension;

        if (STRUCTURE_BY_ITEM.equals(structure)) {
            return subfolder != null
                    ? itemName + "/" + subfolder + "/" + baseName
                    : itemName + "/" + baseName;
        } else {
            return subfolder != null
                    ? subfolder + "/" + baseName
                    : baseName;
        }
    }

    /**
     * 获取文件扩展名
     */
    private String getFileExtension(String fileName) {
        if (fileName == null) return DEFAULT_FILE_EXTENSION;
        int dotIndex = fileName.lastIndexOf('.');
        return dotIndex > 0 ? fileName.substring(dotIndex) : DEFAULT_FILE_EXTENSION;
    }

    /**
     * 创建临时ZIP文件
     */
    protected File createTempZipFile(String zipName) {
        File tempDir = new File(TEMP_DIR);
        if (!tempDir.exists()) {
            tempDir.mkdirs();
        }
        return new File(tempDir, EXPORT_PREFIX + zipName + "_" + IdUtil.simpleUUID() + ".zip");
    }

    /**
     * 上传ZIP文件到存储服务
     */
    protected String uploadZipFile(File zipFile, String zipName) throws Exception {
        String objectName = EXPORT_OBJECT_PREFIX + zipName + "_" + IdUtil.simpleUUID() + ".zip";
        try (FileInputStream fis = new FileInputStream(zipFile)) {
            String url = fileService.uploadFile(objectName, fis, zipFile.length(), ZIP_CONTENT_TYPE);
            log.info("ZIP文件上传成功: url={}", url);
            return url;
        } finally {
            if (zipFile.exists()) {
                zipFile.delete();
            }
        }
    }

    /**
     * 清理临时文件
     */
    protected void cleanupTempFile(File file) {
        if (file != null && file.exists()) {
            if (!file.delete()) {
                log.warn("临时文件删除失败: {}", file.getAbsolutePath());
            }
        }
    }
}

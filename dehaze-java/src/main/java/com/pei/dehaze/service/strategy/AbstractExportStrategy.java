package com.pei.dehaze.service.strategy;

import cn.hutool.core.util.IdUtil;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.form.ExportTaskCreateForm;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.SysFileService;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;

import java.io.File;
import java.io.FileInputStream;
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
public abstract class AbstractExportStrategy implements TaskStrategy {

    protected static final String TEMP_DIR = System.getProperty("java.io.tmpdir") + "export/";
    protected static final String EXPORT_PREFIX = "export_";

    @Resource
    protected SysFileService sysFileService;

    @Resource
    protected FileService fileService;

    /**
     * 从参数中获取导出选项
     */
    protected ExportTaskCreateForm.ExportOptions getExportOptions(Map<String, Object> params) {
        Object optionsObj = params.get("options");
        if (optionsObj instanceof ExportTaskCreateForm.ExportOptions options) {
            return options;
        }
        if (optionsObj instanceof Map<?, ?> optionsMap) {
            ExportTaskCreateForm.ExportOptions options = new ExportTaskCreateForm.ExportOptions();
            Object structureVal = optionsMap.get("structure");
            options.setStructure(structureVal != null ? (String) structureVal : "by_item");
            @SuppressWarnings("unchecked")
            List<String> types = (List<String>) optionsMap.get("includeTypes");
            options.setIncludeTypes(types);
            Object thumbnailVal = optionsMap.get("includeThumbnail");
            options.setIncludeThumbnail(thumbnailVal != null ? (Boolean) thumbnailVal : false);
            return options;
        }
        return new ExportTaskCreateForm.ExportOptions();
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
            byte[] buffer = new byte[8192];
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

        if ("by_item".equals(structure)) {
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
        if (fileName == null) return ".jpg";
        int dotIndex = fileName.lastIndexOf('.');
        return dotIndex > 0 ? fileName.substring(dotIndex) : ".jpg";
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
        String objectName = "exports/" + zipName + "_" + IdUtil.simpleUUID() + ".zip";
        try (FileInputStream fis = new FileInputStream(zipFile)) {
            String url = fileService.uploadFile(objectName, fis, zipFile.length(), "application/zip");
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

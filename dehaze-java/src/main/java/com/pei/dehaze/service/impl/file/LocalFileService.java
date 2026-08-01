package com.pei.dehaze.service.impl.file;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.io.file.PathUtil;
import cn.hutool.core.lang.Assert;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.util.PathSecurityUtil;
import com.pei.dehaze.model.bo.FileBO;
import com.pei.dehaze.service.FileService;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * @author earthy-zinc
 * @since 2024-06-08 19:24:03
 */
@Component
@ConditionalOnProperty(prefix = "file.local", name = "upload-path")
@ConfigurationProperties(prefix = "file.local")
@Data
@Slf4j
public class LocalFileService implements FileService {
    @Value("${file.baseUrl}")
    private String baseUrl;

    private String uploadPath;

    @Override
    public String getStorageType() {
        return "local";
    }

    @Override
    public String getBaseUrl() {
        return baseUrl;
    }

    @Override
    public FileBO uploadFile(FileBO fileBO) {
        Path filePath = Path.of(uploadPath, fileBO.getObjectName());
        Path dirPath = filePath.getParent();
        if (!PathUtil.isDirectory(dirPath) && !PathUtil.exists(dirPath, true)) {
            try {
                Files.createDirectories(dirPath);
            } catch (IOException e) {
                throw new BusinessException("无法为上传文件创建对应的文件夹", e);
            }
        }

        String absolutePath = filePath.toAbsolutePath().toString();

        File file = fileBO.getFile();
        try (FileInputStream stream = new FileInputStream(file)) {
            FileUtil.writeFromStream(stream, absolutePath);
        } catch (IOException e) {
            throw new BusinessException("无法保存文件", e);
        }

        fileBO.setStorage(getStorageType());
        return fileBO;
    }

    @Override
    public String uploadFile(String objectName, InputStream inputStream, long fileSize, String contentType) {
        Assert.notBlank(objectName, "objectName不能为空");
        Assert.notNull(inputStream, "inputStream不能为空");

        Path filePath = Path.of(uploadPath, objectName);
        Path dirPath = filePath.getParent();

        if (!PathUtil.exists(dirPath, true)) {
            try {
                Files.createDirectories(dirPath);
            } catch (IOException e) {
                throw new BusinessException("无法为上传文件创建对应的文件夹: " + e.getMessage(), e);
            }
        }

        try {
            FileUtil.writeFromStream(inputStream, filePath.toAbsolutePath().toString());
            return objectName;
        } catch (Exception e) {
            throw new BusinessException("无法保存文件: " + e.getMessage(), e);
        }
    }


    /**
     * 删除文件
     *
     * @param objectName objectName
     * @return 删除结果
     */
    @Override
    public boolean deleteFile(String objectName) {
        Path filePath = PathSecurityUtil.validatePath(uploadPath, objectName);

        if (!Files.exists(filePath)) {
            log.warn("文件不存在: {}", objectName);
            return true; // 文件不存在视为删除成功（幂等性）
        }

        try {
            Files.delete(filePath);
            log.debug("删除本地文件成功: {}", objectName);
            return true;
        } catch (IOException e) {
            log.error("删除本地文件失败: {}", objectName, e);
            return false;
        }
    }

    @Override
    public InputStream downLoadFile(String objectName) {
        Path filePath = PathSecurityUtil.validatePath(uploadPath, objectName);

        // 验证文件名，避免文件名注入攻击
        PathSecurityUtil.validateFileName(filePath.getFileName().toString());

        if (!Files.exists(filePath)) {
            throw new BusinessException("文件不存在: " + objectName);
        }

        try {
            return new FileInputStream(filePath.toFile());
        } catch (IOException e) {
            throw new BusinessException("文件下载失败: " + e.getMessage(), e);
        }
    }
}

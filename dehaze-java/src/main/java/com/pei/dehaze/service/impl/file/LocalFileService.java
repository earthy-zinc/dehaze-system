package com.pei.dehaze.service.impl.file;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.io.file.PathUtil;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.bo.FileBO;
import com.pei.dehaze.service.FileService;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.Resource;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.ApplicationContext;
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
@ConditionalOnProperty(value = "file.type", havingValue = "local")
@ConfigurationProperties(prefix = "file.local")
@RequiredArgsConstructor
@Data
@Slf4j
public class LocalFileService implements FileService {
    @Value("${file.baseUrl}")
    private String baseUrl;

    private String uploadPath;

    @Resource
    private ApplicationContext applicationContext;

    @Override
    public FileBO uploadFile(FileBO fileBO) {
        Path filePath = Path.of(uploadPath, fileBO.getPath());
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

        fileBO.setPath(fileBO.getPath());

        String url = baseUrl + "/" + fileBO.getObjectName();
        fileBO.setUrl(url);
        return fileBO;
    }


    /**
     * 删除文件
     *
     * @param objectName objectName
     * @return 删除结果
     */
    @Override
    public boolean deleteFile(String objectName) {
        Path filePath = Path.of(uploadPath, objectName);
        // 验证文件路径的安全性，避免路径遍历攻击
        if (!filePath.isAbsolute()) {
            throw new IllegalArgumentException("无效的文件路径");
        }
        if (!Files.exists(filePath)) {
            throw new BusinessException("文件不存在");
        }
        // 目前不让删除本地文件
        return false;
    }

    @Override
    public InputStream downLoadFile(String objectName) {
        Path filePath = Path.of(uploadPath, objectName);
        // 验证文件路径的安全性，避免路径遍历攻击
        if (!filePath.isAbsolute() || !Files.exists(filePath)) {
            throw new IllegalArgumentException("无效的文件路径");
        }

        // 验证文件名，避免文件名注入攻击
        String fileName = filePath.getFileName().toString();
        if (!fileName.matches("[a-zA-Z0-9.\\-_]+")) {
            throw new IllegalArgumentException("不支持的文件名");
        }

        try (FileInputStream fileInputStream = new FileInputStream(filePath.toFile())) {
            return fileInputStream;
        } catch (IOException e) {
            throw new BusinessException("文件下载失败，文件不存在", e);
        }
    }

    @PostConstruct
    public void init() {
    }
}

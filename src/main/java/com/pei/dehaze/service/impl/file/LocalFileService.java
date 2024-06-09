package com.pei.dehaze.service.impl.file;

import cn.hutool.core.io.FileUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.enums.ImageTypeEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.util.FileUploadUtils;
import com.pei.dehaze.model.dto.FileInfo;
import com.pei.dehaze.model.entity.SysImage;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.SysImageService;
import jakarta.annotation.PostConstruct;
import jakarta.servlet.ServletOutputStream;
import jakarta.servlet.http.HttpServletResponse;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import org.apache.tomcat.util.http.fileupload.IOUtils;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * @author earthy-zinc
 * @since 2024-06-08 19:24:03
 */
@Component
@ConditionalOnProperty(value = "file.type", havingValue = "local")
@ConfigurationProperties(prefix = "file.local")
@RequiredArgsConstructor
@Data
public class LocalFileService implements FileService {

    private SysImageService imageService;

    private HttpServletResponse response;
    private String baseUrl;
    private String uploadPath;
    private String datasetPath;
    private String predictPath;

    @Override
    public boolean uploadCheck(String md5) {
        SysImage image = imageService.getOne(new LambdaQueryWrapper<SysImage>().eq(SysImage::getMd5, md5));
        return image == null;
    }

    /**
     * 上传文件
     *
     * @param file 表单文件对象
     * @return 文件信息
     */
    @Override
    @SneakyThrows
    public FileInfo uploadFile(MultipartFile file) {
        if (file != null) {
            try {
                String md5 = FileUploadUtils.getMd5(file.getInputStream());
                SysImage image = imageService.getOne(new LambdaQueryWrapper<SysImage>().eq(SysImage::getMd5, md5));
                FileInfo fileInfo = new FileInfo();
                if (image == null) {
                    String fileExtension = FileUploadUtils.getExtension(file);
                    String fileName = md5 + "." + fileExtension;
                    String filePath = uploadPath + fileName;
                    file.transferTo(new File(filePath));
                    image = new SysImage();
                    image.setUrl(baseUrl + "/upload/" + fileName);
                    image.setType(ImageTypeEnum.UPLOAD.getLabel());
                    image.setSize(FileUtil.readableFileSize(file.getSize()));
                    image.setName(fileName);
                    image.setPath(filePath);
                    image.setMd5(md5);
                    imageService.save(image);
                }
                fileInfo.setName(image.getName());
                fileInfo.setUrl(image.getUrl());
                return fileInfo;
            } catch (IOException e) {
                new BusinessException("文件上传失败");
            }
        }
        return null;
    }

    /**
     * 删除文件
     *
     * @param filePath 文件完整URL
     * @return 删除结果
     */
    @Override
    public boolean deleteFile(String filePath) {
        return false;
    }

    public void download(String filePath) {
        String prefix = filePath.split("/")[0];
        Path path;
        if (prefix.equals(ImageTypeEnum.UPLOAD.getValue())) {
            path = Paths.get(uploadPath, filePath.replaceFirst("upload/", ""));
        } else if (prefix.equals(ImageTypeEnum.DATASET.getValue())) {
            path = Paths.get(datasetPath, filePath.replaceFirst("dataset/", ""));
        } else if (prefix.equals(ImageTypeEnum.PREDICT.getValue())) {
            path = Paths.get(predictPath, filePath.replaceFirst("predict/", ""));
        } else {
            throw new IllegalArgumentException("未找到图片" + filePath);
        }
        downloadFile(path);
    }

    private void downloadFile(Path filePath) {
        // 验证文件路径的安全性，避免路径遍历攻击
        if (!filePath.isAbsolute() || !Files.exists(filePath)) {
            throw new IllegalArgumentException("无效的文件路径");
        }

        // 验证文件名，避免文件名注入攻击
        String fileName = filePath.getFileName().toString();
        if (!fileName.matches("[a-zA-Z0-9.\\-_]+")) {
            throw new IllegalArgumentException("不支持的文件名");
        }

        try {
            // 设置文件名，确保编码正确且避免XSS攻击
            String encodedFileName = URLEncoder.encode(fileName, StandardCharsets.UTF_8).replace("+", "%20");
            response.addHeader("Content-Disposition", "attachment;fileName=" + encodedFileName);

            // 使用try-with-resources语句自动管理资源
            try (FileInputStream fileInputStream = new FileInputStream(filePath.toFile());
                 ServletOutputStream outputStream = response.getOutputStream()) {
                // 复制文件内容
                IOUtils.copyLarge(fileInputStream, outputStream);
            }
        } catch (IOException e) {
            // 记录具体异常信息，提供更详细的错误反馈
            throw new BusinessException("文件下载失败，可能原因是：" + e.getMessage());
        }
    }

    public void updateUrl() {

    }

    @PostConstruct
    public void init() {
        System.out.println("初始化本地文件服务");
    }
}

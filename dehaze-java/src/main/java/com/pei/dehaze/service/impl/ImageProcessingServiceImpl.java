package com.pei.dehaze.service.impl;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.service.ImageProcessingService;
import lombok.extern.slf4j.Slf4j;
import net.coobird.thumbnailator.Thumbnails;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.Iterator;
import java.util.Set;

/**
 * 图片处理服务实现
 *
 * @author earthy-zinc
 * @since 2025-01-19
 */
@Slf4j
@Service
public class ImageProcessingServiceImpl implements ImageProcessingService {

    private static final Set<String> SUPPORTED_FORMATS = Set.of("jpg", "jpeg", "png", "gif", "webp", "bmp");

    @Value("${file.image.max-size:10485760}")
    private long maxFileSize;

    @Value("${file.image.thumbnail.quality:0.5}")
    private float thumbnailQuality;

    @Override
    public void validateImageFile(File file) {
        if (file == null || !file.exists()) {
            throw new BusinessException("文件不能为空");
        }

        if (file.length() > maxFileSize) {
            throw new BusinessException("文件大小不能超过" + (maxFileSize / 1024 / 1024) + "MB");
        }

        String fileName = file.getName().toLowerCase();
        if (!isImage(fileName)) {
            throw new BusinessException("仅支持 " + String.join("/", SUPPORTED_FORMATS) + " 格式");
        }
    }

    @Override
    public void validateImageFile(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new BusinessException("文件不能为空");
        }

        if (file.getSize() > maxFileSize) {
            throw new BusinessException("文件大小不能超过" + (maxFileSize / 1024 / 1024) + "MB");
        }

        String fileName = file.getOriginalFilename();
        if (fileName == null || !isImage(fileName.toLowerCase())) {
            throw new BusinessException("仅支持 " + String.join("/", SUPPORTED_FORMATS) + " 格式");
        }
    }

    @Override
    public File generateThumbnail(File source, int maxWidth) {
        return generateThumbnail(source, maxWidth, maxWidth);
    }

    @Override
    public File generateThumbnail(File source, int width, int height) {
        try {
            File output = Files.createTempFile("tempThumbnail", ".jpg").toFile();
            Thumbnails.of(source)
                    .size(width, height)
                    .outputQuality(thumbnailQuality)
                    .outputFormat("jpg")
                    .toFile(output);
            return output;
        } catch (IOException e) {
            log.error("生成缩略图失败: source={}", source.getAbsolutePath(), e);
            throw new BusinessException("生成缩略图失败", e);
        }
    }

    @Override
    public void generateThumbnail(String srcPath, String destPath, int width, int height) {
        File destDir = new File(destPath).getParentFile();
        if (!destDir.exists() && !destDir.mkdirs()) {
            throw new BusinessException("创建缩略图目录失败: " + destDir.getAbsolutePath());
        }
        try {
            Thumbnails.of(new File(srcPath))
                    .size(width, height)
                    .outputQuality(thumbnailQuality)
                    .toFile(new File(destPath));
        } catch (IOException e) {
            log.error("生成缩略图失败: srcPath={}, destPath={}", srcPath, destPath, e);
            throw new BusinessException("生成缩略图失败", e);
        }
    }

    @Override
    public int[] getImageDimensions(File file) {
        try (InputStream is = new FileInputStream(file)) {
            return getImageDimensions(is);
        } catch (IOException e) {
            throw new BusinessException("解析图片宽高失败: " + file.getAbsolutePath(), e);
        }
    }

    @Override
    public int[] getImageDimensions(MultipartFile file) {
        try (InputStream is = file.getInputStream()) {
            return getImageDimensions(is);
        } catch (IOException e) {
            throw new BusinessException("解析图片宽高失败: " + file.getOriginalFilename(), e);
        }
    }

    private int[] getImageDimensions(InputStream is) throws IOException {
        try (ImageInputStream iis = ImageIO.createImageInputStream(is)) {
            Iterator<ImageReader> readers = ImageIO.getImageReaders(iis);
            if (readers.hasNext()) {
                ImageReader reader = readers.next();
                reader.setInput(iis, true);
                return new int[]{reader.getWidth(0), reader.getHeight(0)};
            }
            throw new IOException("不支持的图片格式");
        }
    }

    @Override
    public boolean isSupportedImageFormat(String extension) {
        if (extension == null) {
            return false;
        }
        return SUPPORTED_FORMATS.contains(extension.toLowerCase());
    }

    @Override
    public boolean isImage(String fileName) {
        if (fileName == null) {
            return false;
        }
        int dotIndex = fileName.lastIndexOf('.');
        if (dotIndex < 0) {
            return false;
        }
        String extension = fileName.substring(dotIndex + 1).toLowerCase();
        return SUPPORTED_FORMATS.contains(extension);
    }

    @Override
    public Set<String> getSupportedFormats() {
        return SUPPORTED_FORMATS;
    }

    @Override
    public long getMaxFileSize() {
        return maxFileSize;
    }
}

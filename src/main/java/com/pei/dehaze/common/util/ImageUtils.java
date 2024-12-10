package com.pei.dehaze.common.util;

import com.pei.dehaze.common.exception.BusinessException;
import lombok.extern.slf4j.Slf4j;
import net.coobird.thumbnailator.Thumbnails;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

@Slf4j
public class ImageUtils {
    public static void generateThumbnail(String srcPath, String destPath, int width, int height) {
       try {
           // 创建目标路径所需的全部目录
           File destDir = new File(destPath).getParentFile();
           synchronized (destDir) {
               if (!destDir.exists() && !destDir.mkdirs()) {
                   throw new BusinessException("创建缩略图目录失败: " + destDir.getAbsolutePath());
               }
           }
           Thumbnails.of(new File(srcPath))
                   .size(width, height)
                   .outputQuality(0.5f)
                   .toFile(new File(destPath));
       } catch (IOException e) {
           throw new BusinessException("生成缩略图失败", e);
       }
    }

    public static File generateThumbnail(File file, int width, int height) {
        try {
            File output = Files.createTempFile("tempThumbnail", ".jpg").toFile();
            Thumbnails.of(file)
                    .size(width, height)      // 设置缩略图的宽度和高度
                    .outputQuality(0.5f)     // 设置输出质量（0.0-1.0）
                    .outputFormat("jpg")      // 设置输出格式（如 jpg）
                    .toFile(output);
            return output;
        } catch (IOException e) {
            throw new BusinessException("生成缩略图失败", e);
        }
    }

    public static boolean isImage(String fileName) {
        return fileName != null && (fileName.endsWith(".jpg") || fileName.endsWith(".png") || fileName.endsWith(".jpeg"));
    }
}

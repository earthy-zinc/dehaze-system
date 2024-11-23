package com.pei.dehaze.common.util;

/**
 * @author earthy-zinc
 * @since 2024-06-08 22:17:57
 */

import cn.hutool.core.io.FileUtil;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.apache.velocity.shaded.commons.io.FilenameUtils;
import org.springframework.util.DigestUtils;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

@Slf4j
public class FileUploadUtils {

    /**
     * 获取文件md5值
     *
     * @param inputStream 文件输入流
     * @return {@link String} 文件md5值
     */
    public static String getMd5(InputStream inputStream) {
        String md5 = null;
        try {
            md5 = DigestUtils.md5DigestAsHex(inputStream);
        } catch (Exception e) {
            log.error("get md5 error, {}", e.getMessage());
        }
        return md5;
    }

    /**
     * 获取文件名的后缀
     *
     * @param file 表单文件
     * @return 后缀名
     */
    public static String getExtension(MultipartFile file) {
        String extension = FilenameUtils.getExtension(file.getOriginalFilename());
        if (StringUtils.isEmpty(extension)) {
            extension = MimeTypeUtils.getExtension(Objects.requireNonNull(file.getContentType()));
        }
        return extension;
    }

    public static int dirFileCount(String dir) {
        File directory = FileUtil.file(dir);
        return dirFileCount(directory);
    }

    public static int dirFileCount(File directory) {
        if (FileUtil.isDirectory(directory)) {
            List<File> files = FileUtil.loopFiles(directory);
            return files.size();
        } else {
            return 0;
        }
    }

    public static int dirFileCount(Path path) {
        File directory = path.toFile();
        return dirFileCount(directory);
    }

    public static String dirSize(String dir) {
        File directory = FileUtil.file(dir);
        return dirSize(directory);
    }

    public static String dirSize(File directory) {
        if (FileUtil.isDirectory(directory)) {
            long size = FileUtil.size(directory);
            return FileUtil.readableFileSize(size);
        } else {
            return "0";
        }
    }

    public static String dirSize(Path path) {
        File directory = path.toFile();
        return dirSize(directory);
    }

    public static String fileSize(String filePath) {
        return FileUtil.readableFileSize(FileUtil.size(new File(filePath)));
    }
}

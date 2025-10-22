package com.pei.dehaze.common.util;


import cn.hutool.core.io.FileUtil;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.bo.DatasetItemBO;
import com.pei.dehaze.model.bo.FileBO;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.apache.velocity.shaded.commons.io.FilenameUtils;
import org.springframework.util.DigestUtils;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * @author earthy-zinc
 * @since 2024-06-08 22:17:57
 */
@Slf4j
public class FileUploadUtils {

    private FileUploadUtils() {
    }

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

    public static FileBO createFileBO(MultipartFile file, String baseUrl, String path) {
        try {
            FileBO fileBO = new FileBO();
            setFileBO(file, baseUrl, path, fileBO);
            return fileBO;
        } catch (IOException e) {
            throw new BusinessException("Error creating fileBO from MultipartFile: " + e.getMessage(), e);
        }
    }

    private static void setFileBO(MultipartFile file, String baseUrl, String path, FileBO fileBO)
            throws IOException {
        InputStream stream = file.getInputStream();
        String filename = file.getOriginalFilename();
        String extension = FileUtil.getSuffix(filename);
        String md5 = FileUploadUtils.getMd5(stream);
        String objectName = path + "/" + md5 + "." + extension;
        String url = baseUrl + "/" + objectName;

        File tempFile = Files.createTempFile(md5, "." + extension).toFile();
        file.transferTo(tempFile);

        fileBO.setFile(tempFile);
        fileBO.setName(filename);
        fileBO.setObjectName(objectName);
        fileBO.setExtension(extension);
        fileBO.setMd5(md5);
        fileBO.setPath(objectName);
        fileBO.setSize(file.getSize());
        fileBO.setUrl(url);
    }

    public static FileBO createFileBO(File file, String baseUrl, String path) {
        try (FileInputStream stream = new FileInputStream(file)) {
            FileBO fileBO = new FileBO();

            String filename = file.getName();
            String extension = FileUtil.getSuffix(filename);
            String md5 = FileUploadUtils.getMd5(stream);
            String objectName = path + "/" + md5 + "." + extension;
            String url = baseUrl + "/" + objectName;

            fileBO.setFile(file);
            fileBO.setName(filename);
            fileBO.setObjectName(objectName);
            fileBO.setExtension(extension);
            fileBO.setMd5(md5);
            fileBO.setPath(objectName);
            fileBO.setSize(file.length());
            fileBO.setUrl(url);
            return fileBO;
        } catch (IOException e) {
            throw new BusinessException("无法创建FileBO", e);
        }
    }

    public static DatasetItemBO createDatasetItemBO(
            MultipartFile file, String baseUrl, String path,
            String type, String description) {
        try {
            DatasetItemBO itemBO = new DatasetItemBO();
            setFileBO(file, baseUrl, path, itemBO);
            itemBO.setType(type);
            itemBO.setDescription(description);
            return itemBO;
        } catch (IOException e) {
            throw new BusinessException("Error creating DatasetItemBO from MultipartFile: " + e.getMessage(), e);
        }
    }
}

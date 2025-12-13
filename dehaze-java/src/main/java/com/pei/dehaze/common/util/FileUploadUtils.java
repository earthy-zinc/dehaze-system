package com.pei.dehaze.common.util;


import cn.hutool.core.io.FileUtil;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.bo.ItemFileBO;
import com.pei.dehaze.model.bo.FileBO;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.apache.velocity.shaded.commons.io.FilenameUtils;
import org.springframework.util.DigestUtils;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
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

    /**
     * 获取图片宽高
     *
     * @param file 图片文件
     * @return int数组，[0]为宽度，[1]为高度，解析失败返回[0,0]
     */
    public static int[] getImageDimensions(MultipartFile file) {
        try (InputStream is = file.getInputStream();
             ImageInputStream iis = ImageIO.createImageInputStream(is)) {
            Iterator<ImageReader> readers = ImageIO.getImageReaders(iis);
            if (readers.hasNext()) {
                ImageReader reader = readers.next();
                reader.setInput(iis, true);
                return new int[]{reader.getWidth(0), reader.getHeight(0)};
            }
        } catch (IOException e) {
            log.warn("解析图片宽高失败: {}", e.getMessage());
        }
        return new int[]{0, 0};
    }

    /**
     * 获取图片宽高
     *
     * @param file 图片文件
     * @return int数组，[0]为宽度，[1]为高度，解析失败返回[0,0]
     */
    public static int[] getImageDimensions(File file) {
        try (InputStream is = new FileInputStream(file);
             ImageInputStream iis = ImageIO.createImageInputStream(is)) {
            Iterator<ImageReader> readers = ImageIO.getImageReaders(iis);
            if (readers.hasNext()) {
                ImageReader reader = readers.next();
                reader.setInput(iis, true);
                return new int[]{reader.getWidth(0), reader.getHeight(0)};
            }
        } catch (IOException e) {
            log.warn("解析图片宽高失败: {}", e.getMessage());
        }
        return new int[]{0, 0};
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

    public static ItemFileBO createItemFileBO(
            MultipartFile file, String baseUrl, String path,
            String type, String description, String sceneType, String hazeLevel) {
        try {
            ItemFileBO itemBO = new ItemFileBO();
            setFileBO(file, baseUrl, path, itemBO);
            itemBO.setType(type);
            itemBO.setDescription(description);
            itemBO.setSceneType(sceneType);
            itemBO.setHazeLevel(hazeLevel);
            // 解析图片宽高
            int[] dimensions = getImageDimensions(file);
            itemBO.setWidth(dimensions[0] > 0 ? dimensions[0] : null);
            itemBO.setHeight(dimensions[1] > 0 ? dimensions[1] : null);

            return itemBO;
        } catch (IOException e) {
            throw new BusinessException("无法从 MultipartFile 创建 ItemFileBO: " + e.getMessage(), e);
        }
    }
    /**
     * 校验图片文件格式和大小
     */
    public static void validateImageFile(File file) {
        if (file == null) {
            throw new BusinessException("文件不能为空");
        }

        // 校验文件大小（10MB限制）
        if (file.length() > 10 * 1024 * 1024) {
            throw new BusinessException("文件大小不能超过10MB");
        }

        // 校验文件格式
        String fileName = file.getName();

        String extension = fileName.toLowerCase();
        if (!extension.endsWith(".jpg") && !extension.endsWith(".jpeg")
                && !extension.endsWith(".png") && !extension.endsWith(".gif")) {
            throw new BusinessException("仅支持 jpg/png/gif 格式");
        }
    }

    /**
     * 校验图片文件格式和大小（MultipartFile版本）
     */
    public static void validateImageFile(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new BusinessException("文件不能为空");
        }

        // 校验文件大小（10MB限制）
        if (file.getSize() > 10 * 1024 * 1024) {
            throw new BusinessException("文件大小不能超过10MB");
        }

        // 校验文件格式
        String fileName = file.getOriginalFilename();
        if (fileName == null) {
            throw new BusinessException("文件名不能为空");
        }

        String extension = fileName.toLowerCase();
        if (!extension.endsWith(".jpg") && !extension.endsWith(".jpeg")
                && !extension.endsWith(".png") && !extension.endsWith(".gif")) {
            throw new BusinessException("仅支持 jpg/png/gif 格式");
        }
    }

}

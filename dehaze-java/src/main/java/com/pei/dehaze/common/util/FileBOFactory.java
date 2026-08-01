package com.pei.dehaze.common.util;

import cn.hutool.core.io.FileUtil;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.bo.FileBO;
import com.pei.dehaze.model.bo.ItemFileBO;
import com.pei.dehaze.service.ImageProcessingService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;

/**
 * FileBO 工厂类
 * 从 FileUploadUtils 抽取的 FileBO 构建逻辑，职责更加单一
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class FileBOFactory {

    private final ImageProcessingService imageProcessingService;

    /**
     * 从 MultipartFile 创建 FileBO
     *
     * @param file 上传的文件
     * @param path 存储路径前缀（如数据集名称）
     * @return FileBO
     */
    public FileBO createFileBO(MultipartFile file, String path) {
        try {
            FileBO fileBO = new FileBO();
            populateFileBO(file, path, fileBO);
            return fileBO;
        } catch (IOException e) {
            throw new BusinessException("创建 FileBO 失败: " + e.getMessage(), e);
        }
    }

    /**
     * 从 File 创建 FileBO
     *
     * @param file 文件对象
     * @param path 存储路径前缀
     * @return FileBO
     */
    public FileBO createFileBO(File file, String path) {
        try (FileInputStream stream = new FileInputStream(file)) {
            FileBO fileBO = new FileBO();

            String filename = file.getName();
            String extension = FileUtil.extName(filename);
            String md5 = FileUploadUtils.getMd5(stream);
            String objectName = path + "/" + md5 + "." + extension;

            fileBO.setFile(file);
            fileBO.setName(filename);
            fileBO.setObjectName(objectName);
            fileBO.setExtension(extension);
            fileBO.setMd5(md5);
            fileBO.setSize(file.length());
            return fileBO;
        } catch (IOException e) {
            throw new BusinessException("创建 FileBO 失败: " + e.getMessage(), e);
        }
    }

    /**
     * 从 MultipartFile 创建 ItemFileBO（数据项图片）
     *
     * @param file        上传的文件
     * @param path        存储路径前缀（如数据集名称）
     * @param type        图片类型（clear/hazy）
     * @param description 描述
     * @param sceneType   场景类型
     * @param hazeLevel   雾霾等级
     * @return ItemFileBO
     */
    public ItemFileBO createItemFileBO(
            MultipartFile file,
            String path,
            String type,
            String description,
            String sceneType,
            String hazeLevel) {
        try {
            ItemFileBO itemBO = new ItemFileBO();
            populateFileBO(file, path, itemBO);
            itemBO.setType(type);
            itemBO.setDescription(description);
            itemBO.setSceneType(sceneType);
            itemBO.setHazeLevel(hazeLevel);

            // 使用已保存的临时文件解析图片宽高（transferTo 后 MultipartFile 已消费）
            int[] dimensions = imageProcessingService.getImageDimensions(itemBO.getFile());
            itemBO.setWidth(dimensions[0] > 0 ? dimensions[0] : null);
            itemBO.setHeight(dimensions[1] > 0 ? dimensions[1] : null);

            return itemBO;
        } catch (IOException e) {
            throw new BusinessException("创建 ItemFileBO 失败: " + e.getMessage(), e);
        }
    }

    /**
     * 填充 FileBO 的公共字段
     */
    private void populateFileBO(MultipartFile file, String path, FileBO fileBO) throws IOException {
        String filename = file.getOriginalFilename();
        String extension = FileUtil.extName(filename);

        // 先 transferTo 保存临时文件（会消费 MultipartFile 的流）
        File tempFile = Files.createTempFile("upload-", "." + extension).toFile();
        file.transferTo(tempFile);

        // 再用临时文件计算 MD5，确保流被正确关闭
        String md5;
        try (FileInputStream stream = new FileInputStream(tempFile)) {
            md5 = FileUploadUtils.getMd5(stream);
        }

        String objectName = path + "/" + md5 + "." + extension;

        fileBO.setFile(tempFile);
        fileBO.setName(filename);
        fileBO.setObjectName(objectName);
        fileBO.setExtension(extension);
        fileBO.setMd5(md5);
        fileBO.setSize(file.getSize());
    }
}

package com.pei.dehaze.common.util;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.util.StrUtil;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.model.dto.FileDTO;
import com.pei.dehaze.model.dto.ItemFileDTO;
import com.pei.dehaze.service.ImageProcessingService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;

/**
 * FileDTO 工厂类
 * 从 FileUploadUtils 抽取的 FileDTO 构建逻辑，职责更加单一
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class FileDTOFactory {

    private final ImageProcessingService imageProcessingService;

    /**
     * 从 MultipartFile 创建 FileDTO
     *
     * @param file 上传的文件
     * @param path 存储路径前缀（如数据集名称）
     * @return FileDTO
     */
    public FileDTO createFileDTO(MultipartFile file, String path) {
        try {
            FileDTO fileDTO = new FileDTO();
            populateFileDTO(file, path, fileDTO);
            return fileDTO;
        } catch (IOException e) {
            throw new BusinessException("创建 FileDTO 失败: " + e.getMessage(), e);
        }
    }

    /**
     * 从 File 创建 FileDTO
     *
     * @param file 文件对象
     * @param path 存储路径前缀
     * @return FileDTO
     */
    public FileDTO createFileDTO(File file, String path) {
        try (FileInputStream stream = new FileInputStream(file)) {
            FileDTO fileDTO = new FileDTO();

            String filename = file.getName();
            String extension = FileUtil.extName(filename);
            String md5 = FileUploadUtils.getMd5(stream);
            String objectName = path + "/" + md5 + "." + extension;

            fileDTO.setFile(file);
            fileDTO.setName(filename);
            fileDTO.setObjectName(objectName);
            fileDTO.setExtension(extension);
            fileDTO.setMd5(md5);
            fileDTO.setSize(file.length());
            return fileDTO;
        } catch (IOException e) {
            throw new BusinessException("创建 FileDTO 失败: " + e.getMessage(), e);
        }
    }

    /**
     * 从 MultipartFile 创建 ItemFileDTO（数据项图片）
     *
     * @param file        上传的文件
     * @param path        存储路径前缀（如数据集名称）
     * @param type        图片类型（clear/hazy）
     * @param description 描述
     * @param sceneType   场景类型
     * @param hazeLevel   雾霾等级
     * @return ItemFileDTO
     */
    public ItemFileDTO createItemFileDTO(
            MultipartFile file,
            String path,
            String type,
            String description,
            String sceneType,
            String hazeLevel) {
        try {
            ItemFileDTO itemBO = new ItemFileDTO();
            populateFileDTO(file, path, itemBO);
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
            throw new BusinessException("创建 ItemFileDTO 失败: " + e.getMessage(), e);
        }
    }

    /**
     * 填充 FileDTO 的公共字段
     */
    private void populateFileDTO(MultipartFile file, String path, FileDTO fileDTO) throws IOException {
        String originalName = file.getOriginalFilename();
        if (StrUtil.isBlank(originalName)) {
            throw new BusinessException("文件名不能为空");
        }
        // basename 化去除路径前缀并校验（三端口径与 Python 端一致）：
        // name 列宽 varchar(100)、拒绝路径分隔符与 :*?"<>| 及控制字符；
        // 反斜杠统一视作路径分隔符（Windows 风格 ..\..\evil 也需剥离）
        String filename = FileUtil.getName(originalName.replace('\\', '/')).trim();
        if (filename.isEmpty() || ".".equals(filename) || "..".equals(filename)) {
            throw new BusinessException("文件名不能为空");
        }
        if (filename.length() > 100) {
            throw new BusinessException("文件名过长");
        }
        if (!filename.matches("[^/:*?\"<>|\\x00-\\x1f]+")) {
            throw new BusinessException("文件名包含非法字符");
        }
        String extension = FileUtil.extName(filename);
        if (extension != null && extension.length() > 20) {
            throw new BusinessException("文件扩展名过长");
        }

        // 先 transferTo 保存临时文件（会消费 MultipartFile 的流）
        File tempFile = Files.createTempFile("upload-", "." + extension).toFile();
        file.transferTo(tempFile);

        // 图片扩展名做文件头魔数校验（防伪装扩展名，三端口径与 Python/Go 一致）
        validateImageMagicBytes(extension, tempFile);

        // 再用临时文件计算 MD5，确保流被正确关闭
        String md5;
        try (FileInputStream stream = new FileInputStream(tempFile)) {
            md5 = FileUploadUtils.getMd5(stream);
        }

        String objectName = path + "/" + md5 + "." + extension;

        fileDTO.setFile(tempFile);
        fileDTO.setName(filename);
        fileDTO.setObjectName(objectName);
        fileDTO.setExtension(extension);
        fileDTO.setMd5(md5);
        fileDTO.setSize(file.getSize());
    }

    /**
     * 图片扩展名需与文件头魔数一致，防止伪装扩展名上传恶意内容。
     * 仅图片类型（jpg/jpeg/png/gif/bmp/webp）校验，完整 MIME 嗅探规划中。
     */
    private void validateImageMagicBytes(String extension, File tempFile) {
        String ext = extension == null ? "" : extension.toLowerCase();
        boolean isImage = switch (ext) {
            case "jpg", "jpeg", "png", "gif", "bmp", "webp" -> true;
            default -> false;
        };
        if (!isImage) {
            return;
        }
        byte[] head;
        try (InputStream in = new FileInputStream(tempFile)) {
            head = in.readNBytes(12);
        } catch (IOException e) {
            throw new BusinessException("读取文件头失败", e);
        }
        boolean ok = switch (ext) {
            case "jpg", "jpeg" -> startsWith(head, new byte[]{(byte) 0xFF, (byte) 0xD8, (byte) 0xFF});
            case "png" -> startsWith(head, new byte[]{(byte) 0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'});
            case "gif" -> startsWith(head, "GIF87a".getBytes()) || startsWith(head, "GIF89a".getBytes());
            case "bmp" -> startsWith(head, new byte[]{'B', 'M'});
            case "webp" -> head.length >= 12 && startsWith(head, "RIFF".getBytes())
                    && head[8] == 'W' && head[9] == 'E' && head[10] == 'B' && head[11] == 'P';
            default -> false;
        };
        if (!ok) {
            throw new BusinessException(ResultCode.FILE_TYPE_NOT_SUPPORTED, "文件内容与图片类型不符");
        }
    }

    private static boolean startsWith(byte[] data, byte[] prefix) {
        if (data.length < prefix.length) {
            return false;
        }
        for (int i = 0; i < prefix.length; i++) {
            if (data[i] != prefix[i]) {
                return false;
            }
        }
        return true;
    }
}

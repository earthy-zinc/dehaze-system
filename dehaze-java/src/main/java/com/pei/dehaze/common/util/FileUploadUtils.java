package com.pei.dehaze.common.util;


import cn.hutool.core.io.FileUtil;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.dto.FileDTO;
import lombok.extern.slf4j.Slf4j;
import org.springframework.util.DigestUtils;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

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
     * @throws BusinessException 当 MD5 计算失败时抛出，避免返回 null 导致下游路径拼接错误
     */
    public static String getMd5(InputStream inputStream) {
        try {
            return DigestUtils.md5DigestAsHex(inputStream);
        } catch (IOException e) {
            throw new BusinessException("计算文件MD5失败", e);
        }
    }

    public static String fileSize(String filePath) {
        return FileUtil.readableFileSize(FileUtil.size(new File(filePath)));
    }

    public static FileDTO createFileDTO(File file, String path) {
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
            throw new BusinessException("无法创建FileDTO", e);
        }
    }

}

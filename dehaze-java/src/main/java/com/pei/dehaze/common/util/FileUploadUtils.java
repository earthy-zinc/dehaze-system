package com.pei.dehaze.common.util;


import cn.hutool.core.io.FileUtil;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.bo.FileBO;
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

    public static FileBO createFileBO(File file, String baseUrl, String path) {
        try (FileInputStream stream = new FileInputStream(file)) {
            FileBO fileBO = new FileBO();

            String filename = file.getName();
            String extension = FileUtil.extName(filename);
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

}

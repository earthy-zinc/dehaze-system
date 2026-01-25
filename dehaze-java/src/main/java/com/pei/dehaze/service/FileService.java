package com.pei.dehaze.service;

import com.pei.dehaze.model.bo.FileBO;

import java.io.InputStream;

/**
 * 对象存储服务接口层
 *
 * @author earthyzinc
 * @since 2022/11/19
 */
public interface FileService {
    /**
     *
     * @param fileBO 文件包装类
     * @return fileUrl 文件访问路径
     */
    FileBO uploadFile(FileBO fileBO);

    /**
     * 上传文件（使用输入流）
     *
     * @param objectName  文件对象名
     * @param inputStream 文件输入流
     * @param fileSize    文件大小
     * @param contentType 文件类型
     * @return 文件访问路径
     */
    String uploadFile(String objectName, InputStream inputStream, long fileSize, String contentType);

    /**
     * 删除文件
     *
     * @param objectName 文件完整 objectName
     * @return 删除结果
     */
    boolean deleteFile(String objectName);

    /**
     * 根据 objectName 下载文件
     */
    InputStream downLoadFile(String objectName);
}

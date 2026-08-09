package com.pei.dehaze.service;

import com.pei.dehaze.model.dto.FileDTO;

import java.io.InputStream;

/**
 * 存储后端服务接口。
 * 每个实现对应一种 storage 标识（minio / local / nginx-static），负责上传/下载 IO 与 baseUrl 配置。
 * URL 永远由 {@link #getUrl(String)} 在运行时拼接，不落库。
 *
 * @author earthyzinc
 * @since 2022/11/19
 */
public interface FileService {

    /**
     * 返回该后端的 storage 标识，用于 sys_file.storage 字段与路由选择。
     */
    String getStorageType();

    /**
     * 返回该后端对外访问的 baseUrl（必须带 scheme + host，例如 https://cdn.example.com）。
     */
    String getBaseUrl();

    /**
     * 按 objectName 拼接完整可访问 URL（baseUrl + "/" + objectName）。
     * 必须返回完整地址，禁止相对路径。
     */
    default String getUrl(String objectName) {
        return getBaseUrl() + "/" + objectName;
    }

    /**
     * 上传文件（基于 FileDTO）。
     *
     * @param fileDTO 文件包装类
     * @return 上传完成后的 FileDTO（不设置 url，仅写 objectName + storage）
     */
    FileDTO uploadFile(FileDTO fileDTO);

    /**
     * 上传文件（使用输入流）。
     *
     * @param objectName  文件对象名
     * @param inputStream 文件输入流
     * @param fileSize    文件大小
     * @param contentType 文件类型
     * @return 文件 objectName（用于落库 sys_file.object_name）
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
     * 根据 objectName 下载文件流。
     */
    InputStream downLoadFile(String objectName);
}

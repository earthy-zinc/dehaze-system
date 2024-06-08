package com.pei.dehaze.service;

import com.pei.dehaze.model.dto.FileInfo;
import org.springframework.web.multipart.MultipartFile;

/**
 * 对象存储服务接口层
 *
 * @author earthyzinc
 * @since 2022/11/19
 */
public interface FileService {
    /**
     * 文件上传检查
     *
     * @param md5
     * @return true 表示文件已存在
     */
    boolean uploadCheck(String md5);

    /**
     * 上传文件
     * @param file 表单文件对象
     * @return 文件信息
     */
    FileInfo uploadFile(MultipartFile file);

    /**
     * 删除文件
     *
     * @param filePath 文件完整URL
     * @return 删除结果
     */
    boolean deleteFile(String filePath);


}

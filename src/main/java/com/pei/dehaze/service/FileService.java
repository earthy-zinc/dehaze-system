package com.pei.dehaze.service;

import com.pei.dehaze.model.dto.ImageFileInfo;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.form.ImageForm;
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
    SysFile uploadFile(MultipartFile file);

    /**
     *
     * @param oldFile 源文件信息
     * @param modelId 模型id
     * @return file
     */
    SysFile getWpxFile(SysFile oldFile, Long modelId);

    /**
     * 上传图片
     *
     * @param file      表单文件对象
     * @param imageForm 图片表单
     * @return 文件信息
     */
    ImageFileInfo uploadImage(MultipartFile file, ImageForm imageForm);

    /**
     * 删除文件
     *
     * @param filePath 文件完整URL
     * @return 删除结果
     */
    boolean deleteFile(String filePath);

    /**
     * 删除图片
     *
     * @param filePath 当前图片在数据库 SysFile 数据表中的存储 URL
     * @return
     */
    boolean deleteImage(String filePath);

}

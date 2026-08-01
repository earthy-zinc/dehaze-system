package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.bo.FileBO;
import com.pei.dehaze.model.entity.SysFile;

import java.io.InputStream;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:34:32
 */
public interface SysFileService extends IService<SysFile> {

    /**
     * 根据md5查询文件，存在则返回文件信息，否则返回null
     *
     * @param md5 文件md5
     * @return 文件信息，不存在返回null
     */
    SysFile check(String md5);

    /**
     * 保存文件，并记录到数据库中
     *
     * @param fileBO
     * @return
     */
    SysFile saveFile(FileBO fileBO);

    /**
     * 仅创建文件记录（不上传到对象存储），用于 nginx 直服的数据集文件
     * MD5 去重：已存在则直接返回
     *
     * @param fileBO 文件信息（objectName + storage 已设置）
     * @return 文件记录
     */
    SysFile saveFileRecord(FileBO fileBO);

    /**
     * @param oldFile 源文件信息
     * @param modelId 模型id
     * @return file
     */
    SysFile getWpxFile(SysFile oldFile, Long modelId);

    boolean deleteFile(Long fileId);

    InputStream download(String objectName);

    /**
     * 填充运行时拼接的访问 URL（不落库）
     */
    void fillUrl(SysFile file);
}

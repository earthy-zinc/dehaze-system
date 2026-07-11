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
     * @param oldFile 源文件信息
     * @param modelId 模型id
     * @return file
     */
    SysFile getWpxFile(SysFile oldFile, Long modelId);

    boolean deleteFile(Long fileId);

    InputStream download(String objectName);
}

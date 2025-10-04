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

    boolean check(String md5);

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

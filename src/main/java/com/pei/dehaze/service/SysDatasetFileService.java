package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysDatasetFile;

public interface SysDatasetFileService extends IService<SysDatasetFile> {
    /**
     * 获取最大的id
     *
     * @return
     */
    Long getMaxImageItemId();
}

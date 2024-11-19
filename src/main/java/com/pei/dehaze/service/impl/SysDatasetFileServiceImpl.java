package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.mapper.SysDatasetFileMapper;
import com.pei.dehaze.model.entity.SysDatasetFile;
import com.pei.dehaze.service.SysDatasetFileService;

public class SysDatasetFileServiceImpl extends ServiceImpl<SysDatasetFileMapper, SysDatasetFile>
        implements SysDatasetFileService {
    @Override
    public Long getMaxImageItemId() {
        QueryWrapper<SysDatasetFile> queryWrapper = new QueryWrapper<>();
        queryWrapper.select("MAX(image_item_id) as maxImageItemId");
        SysDatasetFile result = this.getOne(queryWrapper);
        return result != null ? result.getImageItemId() : 0L;
    }
}

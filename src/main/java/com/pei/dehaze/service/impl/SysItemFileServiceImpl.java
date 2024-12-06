package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.mapper.SysItemFileMapper;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.service.SysItemFileService;
import org.springframework.stereotype.Service;

@Service
public class SysItemFileServiceImpl extends ServiceImpl<SysItemFileMapper, SysItemFile>
        implements SysItemFileService {
}

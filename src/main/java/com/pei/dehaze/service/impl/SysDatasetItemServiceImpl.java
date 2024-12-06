package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.mapper.SysDatasetItemMapper;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.service.SysDatasetItemService;
import org.springframework.stereotype.Service;

@Service
public class SysDatasetItemServiceImpl extends ServiceImpl<SysDatasetItemMapper, SysDatasetItem>
        implements SysDatasetItemService {
}

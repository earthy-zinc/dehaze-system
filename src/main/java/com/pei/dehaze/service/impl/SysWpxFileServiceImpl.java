package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.mapper.SysWpxFileMapper;
import com.pei.dehaze.model.entity.SysWpxFile;
import com.pei.dehaze.service.SysWpxFileService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class SysWpxFileServiceImpl extends ServiceImpl<SysWpxFileMapper, SysWpxFile> implements SysWpxFileService {
}

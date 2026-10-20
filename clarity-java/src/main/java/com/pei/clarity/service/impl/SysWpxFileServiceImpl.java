package com.pei.clarity.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.clarity.mapper.SysWpxFileMapper;
import com.pei.clarity.model.entity.SysWpxFile;
import com.pei.clarity.service.SysWpxFileService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class SysWpxFileServiceImpl extends ServiceImpl<SysWpxFileMapper, SysWpxFile> implements SysWpxFileService {}

package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.mapper.SysEvalLogMapper;
import com.pei.dehaze.model.entity.SysEvalLog;
import com.pei.dehaze.service.SysEvalLogService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class SysEvalLogServiceImpl extends ServiceImpl<SysEvalLogMapper, SysEvalLog> implements SysEvalLogService {
}

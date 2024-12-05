package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.model.entity.SysPredLog;
import com.pei.dehaze.service.SysPredLogService;

@Service
@RequiredArgsConstructor
public class SysPredLogServiceImpl extends ServiceImpl<SysPredLogMapper, SysPredLog> implements SysPredLogService {
}

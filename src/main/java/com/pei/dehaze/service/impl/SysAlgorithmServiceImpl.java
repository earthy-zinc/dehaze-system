package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.mapper.SysAlgorithmMapper;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.query.AlgorithmQuery;
import com.pei.dehaze.model.vo.AlgorithmVO;
import com.pei.dehaze.service.SysAlgorithmService;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:35:44
 */
@Service
public class SysAlgorithmServiceImpl extends ServiceImpl<SysAlgorithmMapper, SysAlgorithm> implements SysAlgorithmService {
    @Override
    public List<AlgorithmVO> getList(AlgorithmQuery queryParams) {
        return null;
    }

    @Override
    public List<Option<?>> getOption() {
        return null;
    }
}

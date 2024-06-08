package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.mapper.SysDatasetMapper;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.query.DatasetQuery;
import com.pei.dehaze.model.vo.DatasetVO;
import com.pei.dehaze.model.vo.ImageItemVO;
import com.pei.dehaze.service.SysDatasetService;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:37:17
 */
@Service
public class SysDatasetServiceImpl extends ServiceImpl<SysDatasetMapper, SysDataset> implements SysDatasetService {
    @Override
    public List<DatasetVO> getList(DatasetQuery queryParams) {
        return null;
    }

    @Override
    public List<ImageItemVO> getImageItem(Long id) {
        return null;
    }

    @Override
    public void deleteDatasets(List<Long> ids) {

    }
}

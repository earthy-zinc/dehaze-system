package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.common.base.BasePageQuery;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.query.DatasetQuery;
import com.pei.dehaze.model.vo.DatasetVO;
import com.pei.dehaze.model.vo.ImageItemVO;
import org.springframework.data.domain.Page;

import java.util.List;

public interface SysDatasetService extends IService<SysDataset> {
    List<DatasetVO> getList(DatasetQuery queryParams);

    Page<ImageItemVO> getImageItem(Long id, BasePageQuery pageQuery);

    boolean deleteDatasets(List<Long> ids);
}

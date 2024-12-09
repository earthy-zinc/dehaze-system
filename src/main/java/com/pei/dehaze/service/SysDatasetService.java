package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.form.DatasetForm;
import com.pei.dehaze.model.query.DatasetQuery;
import com.pei.dehaze.model.vo.DatasetVO;

import java.util.List;

public interface SysDatasetService extends IService<SysDataset> {
    List<DatasetVO> getList(DatasetQuery queryParams);

    boolean addDataset(DatasetForm dataset);

    boolean updateDataset(DatasetForm dataset);

    boolean deleteDatasets(List<Long> ids);

    List<Option<Long>> getOptions();

    List<Long> getLeafDatasetIds();

    List<Long> getLeafDatasetId(Long id);

    SysDataset getRootDataset(Long id);

    SysDataset getDatasetById(Long id);
}

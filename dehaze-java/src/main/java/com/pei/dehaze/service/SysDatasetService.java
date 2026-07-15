package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.model.dto.DatasetStatistics;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.form.DatasetAddForm;
import com.pei.dehaze.model.form.DatasetUpdateForm;
import com.pei.dehaze.model.query.DatasetQuery;
import com.pei.dehaze.model.vo.DatasetVO;

import java.util.List;
import java.util.Map;

public interface SysDatasetService extends IService<SysDataset> {

    IPage<DatasetVO> listPagedDatasets(DatasetQuery queryParams);

    List<DatasetVO> listChildren(Long parentId);

    DatasetVO addDataset(DatasetAddForm dataset);

    DatasetVO updateDataset(Long id, DatasetUpdateForm form);

    List<Option<Long>> getOptions();

    List<Long> getLeafDatasetIds();

    List<Long> getLeafDatasetId(Long id);

    List<Long> getDatasetAndDescendantIds(Long datasetId);

    SysDataset getRootDataset(Long id);

    SysDataset getSysDatasetById(Long id);

    DatasetVO getDatasetById(Long id);

    void deleteDataset(Long id);

    void incrementUsageCount(Long id);

    List<SysItemFile> getDatasetImages(Long datasetId, boolean recursive);

    String getDatasetNameByItemId(Long itemId);

    void evictAllDatasetsCache();

    DatasetStatistics calculateStatistics(Long datasetId);

    List<SysDataset> getAllDatasets();

    Map<Long, DatasetStatistics> getAllDatasetStats();
}

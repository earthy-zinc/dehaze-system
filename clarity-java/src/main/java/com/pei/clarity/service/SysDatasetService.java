package com.pei.clarity.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.clarity.common.model.Option;
import com.pei.clarity.model.dto.DatasetStatistics;
import com.pei.clarity.model.entity.SysDataset;
import com.pei.clarity.model.entity.SysItemFile;
import com.pei.clarity.model.form.DatasetAddForm;
import com.pei.clarity.model.form.DatasetUpdateForm;
import com.pei.clarity.model.query.DatasetQuery;
import com.pei.clarity.model.vo.DatasetVO;
import java.util.List;
import java.util.Map;

public interface SysDatasetService extends IService<SysDataset> {

    /** 测试集选项查询（评估接入）：含清晰图 GT 且启用的数据集，按 taskType 过滤 */
    java.util.List<java.util.Map<String, Object>> getEvaluationOptions(String taskType);

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

    void incrementUsageCount(Long id);

    List<SysItemFile> getDatasetImages(Long datasetId, boolean recursive);

    String getDatasetNameByItemId(Long itemId);

    void evictAllDatasetsCache();

    void evictDatasetMetadataCache();

    DatasetStatistics calculateStatistics(Long datasetId);

    List<SysDataset> getAllDatasets();

    Map<Long, DatasetStatistics> getAllDatasetStats();
}

package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysItemFile;
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

    /**
     * 获取数据集详情（包含统计信息）
     *
     * @param id 数据集ID
     * @return 数据集VO（含统计信息）
     */
    DatasetVO getDatasetDetail(Long id);

    /**
     * 增加数据集使用次数
     *
     * @param id 数据集ID
     */
    void incrementUsageCount(Long id);

    /**
     * 获取数据集中所有图片文件列表（用于下载）
     *
     * @param datasetId 数据集ID
     * @param recursive 是否递归获取子数据集
     * @return 图片文件列表
     */
    List<SysItemFile> getDatasetImages(Long datasetId, boolean recursive);
}

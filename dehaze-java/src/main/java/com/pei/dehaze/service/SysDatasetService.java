package com.pei.dehaze.service;

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

public interface SysDatasetService extends IService<SysDataset> {
    List<DatasetVO> getList(DatasetQuery queryParams);

    DatasetVO addDataset(DatasetAddForm dataset);

    /**
     * 更新数据集
     *
     * @param id   数据集ID
     * @param form 更新表单
     * @return 更新后的数据集VO
     */
    DatasetVO updateDataset(Long id, DatasetUpdateForm form);

    List<Option<Long>> getOptions();

    List<Long> getLeafDatasetIds();

    List<Long> getLeafDatasetId(Long id);

    /**
     * 获取数据集及其所有子孙数据集的ID列表
     *
     * @param datasetId 数据集ID
     * @return ID列表（包含自身）
     */
    List<Long> getDatasetAndDescendantIds(Long datasetId);

    SysDataset getRootDataset(Long id);

    SysDataset getSysDatasetById(Long id);

    /**
     * 获取数据集详情（包含统计信息）
     *
     * @param id 数据集ID
     * @return 数据集VO（含统计信息）
     */
    DatasetVO getDatasetById(Long id);

    /**
     * 删除单个数据集
     *
     * @param id 数据集ID
     */
    void deleteDataset(Long id);

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

    /**
     * 根据数据项ID获取所属数据集名称
     *
     * @param itemId 数据项ID
     * @return 数据集名称
     */
    String getDatasetNameByItemId(Long itemId);

    /**
     * 清除指定数据集的统计缓存
     *
     * @param datasetId 数据集ID
     */
    void evictDatasetStatsCache(Long datasetId);

    /**
     * 清除数据集及其所有祖先的统计缓存
     *
     * @param datasetId 数据集ID
     */
    void evictDatasetAndAncestorStatsCache(Long datasetId);

    /**
     * 计算数据集统计信息（支持缓存）
     *
     * @param datasetId 数据集ID
     * @return 统计信息
     */
    DatasetStatistics calculateStatistics(Long datasetId);
}

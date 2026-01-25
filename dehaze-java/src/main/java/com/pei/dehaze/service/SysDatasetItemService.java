package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.query.DatasetItemQuery;
import com.pei.dehaze.model.vo.BatchOperationResultVO;
import com.pei.dehaze.model.vo.DatasetItemVO;

import java.util.List;

public interface SysDatasetItemService extends IService<SysDatasetItem> {
    SysDatasetItem createDatasetItem(Long datasetId);
    SysDatasetItem createDatasetItem(Long datasetId, String itemName);

    /**
     * 创建数据项并返回完整的VO对象
     *
     * @param datasetId 数据集ID
     * @param itemName  数据项名称
     * @return 数据项VO
     */
    DatasetItemVO createAndReturnDatasetItem(Long datasetId, String itemName);

    void deleteDatasetItem(Long datasetItemId);
    void updateDatasetItem(Long datasetItemId, String itemName);

    /**
     * 修改数据项并返回完整的VO对象
     *
     * @param datasetItemId 数据项ID
     * @param itemName      数据项名称
     * @param sceneType     场景类型
     * @return 数据项VO
     */
    DatasetItemVO updateAndReturnDatasetItem(Long datasetItemId, String itemName, String sceneType);

    /**
     * 搜索图片
     *
     * @param query 查询参数
     * @return 分页结果
     */
    Page<DatasetItemVO> pageSearchDatasetItems(DatasetItemQuery query);

    /**
     * 获取数据项详情
     *
     * @param id 数据项ID
     * @return 数据项详情VO，包含配对图片信息
     */
    DatasetItemVO getDatasetItem(Long id);

    /**
     * 根据数据项ID获取所属数据集ID
     *
     * @param itemId 数据项ID
     * @return 数据集ID，如果不存在返回null
     */
    Long getDatasetIdByItemId(Long itemId);
}

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
    void deleteDatasetItem(Long datasetItemId);
    void updateDatasetItem(Long datasetItemId, String itemName);

    /**
     * 搜索图片
     *
     * @param query 查询参数
     * @return 分页结果
     */
    Page<DatasetItemVO> pageSearchDatasetItems(DatasetItemQuery query);

    /**
     * 批量删除数据项
     *
     * @param itemIds 数据项ID列表
     * @return 删除结果（成功数量，失败数量）
     */
    BatchOperationResultVO batchDeleteDatasetItems(List<Long> itemIds);

    /**
     * 获取数据项详情
     *
     * @param id 数据项ID
     * @return 数据项详情VO，包含配对图片信息
     */
    DatasetItemVO getDatasetItem(Long id);
}

package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.vo.ImageItemVO;

public interface SysDatasetItemService extends IService<SysDatasetItem> {
    SysDatasetItem createDatasetItem(Long datasetId);
    SysDatasetItem createDatasetItem(Long datasetId, String itemName);
    void deleteDatasetItem(Long datasetItemId);
    void updateDatasetItem(Long datasetItemId, String itemName);
    Page<ImageItemVO> getPagedImageItemVOs(Long datasetId, int pageNum, int pageSize);
}

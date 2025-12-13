package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.bo.ItemFileBO;
import com.pei.dehaze.model.dto.ImageFileInfo;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.form.BatchDatasetItemUploadForm;
import com.pei.dehaze.model.form.ImageItemForm;
import com.pei.dehaze.model.form.DatasetItemUploadForm;
import com.pei.dehaze.model.vo.*;

import java.util.List;
import java.util.Map;

public interface SysItemFileService extends IService<SysItemFile> {
    ImageFileInfo saveItemFile(Long itemId, ItemFileBO itemBO);

    List<ImageUrlVO> getImageUrlVOs(Long itemId);

    boolean deleteItemFile(Long itemId);

    /**
     * 获取图片详情
     *
     * @param id ItemFile ID
     * @return 图片详情VO
     */
    ImageDetailVO getImageDetail(Long id);

    /**
     * 修改图片信息（包含标注信息）
     * 合并了原来的"修改图片信息"和"图片标注"功能
     *
     * @param form 图片信息表单
     * @return 更新结果
     */
    boolean updateImageItemInfo(ImageItemForm form);

    /**
     * 增加图片使用次数
     *
     * @param id ItemFile ID
     */
    void incrementUsageCount(Long id);

    /**
     * 保存配对图片（一张清晰图+多张有雾图）
     *
     * @param form 配对上传表单
     * @return 配对结果
     */
    DatasetItemVO createDatasetItemAndUpload(DatasetItemUploadForm form);

    /**
     * 批量保存配对图片（按文件名自动匹配）
     *
     * @param form 批量上传表单
     * @return 批量处理结果
     */
    BatchUploadResultVO batchCreateDatasetItemAndUpload(BatchDatasetItemUploadForm form);
}

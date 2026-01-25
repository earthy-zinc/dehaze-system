package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.bo.ItemFileBO;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.form.BatchDeleteForm;
import com.pei.dehaze.model.form.ItemFileUpdateForm;
import com.pei.dehaze.model.vo.BatchDeleteResultVO;
import com.pei.dehaze.model.vo.ImageUrlVO;

import java.util.List;

public interface SysItemFileService extends IService<SysItemFile> {

    /**
     * 保存数据项图片
     *
     * @param itemId 所属数据项ID
     * @param itemBO 图片业务对象
     * @return 图片信息VO
     */
    ImageUrlVO saveItemFile(Long itemId, ItemFileBO itemBO);

    /**
     * 获取指定数据项的图片列表
     *
     * @param itemId 数据项ID
     * @return 图片URL列表
     */
    List<ImageUrlVO> getImageUrlVOs(Long itemId);

    /**
     * 删除图片
     *
     * @param id 图片ID
     * @return 是否删除成功
     */
    boolean deleteFile(Long id);

    /**
     * 批量删除图片
     *
     * @param ids 图片ID列表
     * @return 批量删除结果
     */
    BatchDeleteResultVO batchDelete(List<Long> ids);

    /**
     * 获取图片详情（包含配对图片和数据项信息）
     *
     * @param id 图片ID
     * @return 图片详情VO（包含配对图片列表和数据项信息）
     */
    ImageUrlVO getImageById(Long id);

    /**
     * 修改图片信息（包含标注信息）
     * 合并了原来的"修改图片信息"和"图片标注"功能
     *
     * @param id 图片ID
     * @param form 图片更新表单
     * @return 更新结果
     */
    boolean updateItemFileInfo(Long id, ItemFileUpdateForm form);

    /**
     * 增加图片使用次数
     *
     * @param id 图片ID
     */
    void incrementUsageCount(Long id);

    /**
     * 将SysItemFile实体转换为ImageUrlVO
     * 用于批量查询时避免N+1问题
     *
     * @param itemFile 数据项文件实体
     * @return 图片URL信息VO
     */
    ImageUrlVO convertToImageUrlVO(SysItemFile itemFile);
}

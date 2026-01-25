package com.pei.dehaze.service;

import com.pei.dehaze.model.form.BatchDatasetItemUploadForm;
import com.pei.dehaze.model.form.DatasetItemUploadForm;
import com.pei.dehaze.model.vo.BatchDeleteResult;
import com.pei.dehaze.model.vo.BatchOperationResultVO;
import com.pei.dehaze.model.vo.BatchUploadResultVO;
import com.pei.dehaze.model.vo.DatasetItemVO;

import java.util.List;

/**
 * 数据集操作服务
 * 处理跨服务的复杂组合操作，避免循环依赖
 *
 * @author earthy-zinc
 * @since 2025-12-13
 */
public interface DatasetOperationService {

    /**
     * 创建数据项并上传配对图片
     * 将创建DatasetItem和上传ItemFile的逻辑组合在一起
     *
     * @param form 配对上传表单
     * @return 创建的数据项VO
     */
    DatasetItemVO createDatasetItemWithImages(DatasetItemUploadForm form);

    /**
     * 批量创建数据项并上传配对图片
     * 根据图片名称，自动对同一有雾/无雾图片进行配对组成数据项
     *
     * @param form 批量上传表单
     * @return 批量处理结果
     */
    BatchUploadResultVO batchCreateDatasetItemsWithImages(BatchDatasetItemUploadForm form);

    /**
     * 级联删除数据项（包括关联的图片文件）
     *
     * @param datasetItemId 数据项ID
     */
    void deleteDatasetItemCascade(Long datasetItemId);

    /**
     * 批量级联删除数据项
     *
     * @param datasetItemIds 数据项ID列表
     */
    void batchDeleteDatasetItemsCascade(List<Long> datasetItemIds);

    /**
     * 批量级联删除数据项（带返回结果）
     *
     * @param datasetItemIds 数据项ID列表
     * @return 批量操作结果
     */
    BatchOperationResultVO batchDeleteDatasetItemsCascadeWithResult(List<Long> datasetItemIds);

    /**
     * 级联删除数据集（包括子数据集、数据项、图片文件）
     *
     * @param datasetIds 数据集ID列表
     * @return 批量删除结果
     */
    BatchDeleteResult batchDeleteDatasets(List<Long> datasetIds);
}

package com.pei.dehaze.service;

import com.pei.dehaze.model.vo.DownloadTaskVO;

import java.util.List;

/**
 * 下载服务接口
 *
 * @author earthy-zinc
 * @since 2025-12-07
 */
public interface DownloadService {

    /**
     * 创建数据集下载任务
     *
     * @param datasetId      数据集ID
     * @param organizeByItem 是否按数据项分目录组织
     * @return 任务ID
     */
    String createDatasetDownloadTask(Long datasetId, boolean organizeByItem);

    /**
     * 创建批量图片下载任务
     *
     * @param itemFileIds    图片ID列表
     * @param organizeByItem 是否按数据项分目录组织
     * @return 任务ID
     */
    String createBatchImageItemDownloadTask(List<Long> itemFileIds, boolean organizeByItem);

    /**
     * 异步处理数据集下载任务
     *
     * @param taskId         任务ID
     * @param datasetId      数据集ID
     * @param organizeByItem 是否按数据项分目录组织
     */
    void processDatasetDownloadTask(String taskId, Long datasetId, boolean organizeByItem);

    /**
     * 异步处理批量图片下载任务
     *
     * @param taskId         任务ID
     * @param itemFileIds    图片ID列表
     * @param organizeByItem 是否按数据项分目录组织
     */
    void processBatchImageDownloadTask(String taskId, List<Long> itemFileIds, boolean organizeByItem);

    /**
     * 获取下载任务状态
     *
     * @param taskId 任务ID
     * @return 下载任务信息
     */
    DownloadTaskVO getTaskStatus(String taskId);
}

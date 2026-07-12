package com.pei.dehaze.sdk.model.dataset;

import lombok.Data;

import java.util.Map;

/**
 * 数据集统计信息（对齐后端 DatasetStatistics）
 */
@Data
public class DatasetStatistics {
    private Long itemCount;
    private Long fileCount;
    private Long totalSize;
    private Long clearCount;
    private Long hazyCount;
    private Map<String, Long> sceneDistribution;
    private Map<String, Long> hazeDistribution;
    private Map<String, Long> formatDistribution;
}

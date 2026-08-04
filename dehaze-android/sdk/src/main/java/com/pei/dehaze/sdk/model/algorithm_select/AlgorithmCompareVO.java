package com.pei.dehaze.sdk.model.algorithm_select;

import lombok.Data;

/**
 * 算法对比结果项
 * 对齐后端 AlgorithmCompareVO（/api/v1/algorithms/select/compare）
 */
@Data
public class AlgorithmCompareVO {
    /** 算法ID */
    private Long algorithmId;
    /** 算法名称 */
    private String algorithmName;
    /** 处理结果URL */
    private String resultUrl;
    /** 处理耗时(毫秒) */
    private Integer time;
    /** 评估指标（PSNR/SSIM等，JSON字符串） */
    private String metrics;
}

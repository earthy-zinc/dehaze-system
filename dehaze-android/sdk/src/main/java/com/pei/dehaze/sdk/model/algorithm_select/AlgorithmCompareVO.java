package com.pei.dehaze.sdk.model.algorithm_select;

import lombok.Data;

/**
 * 算法对比结果VO
 * 对齐后端 AlgorithmCompareVO
 */
@Data
public class AlgorithmCompareVO {
    /** 算法ID */
    private long algorithmId;
    /** 算法名称 */
    private String algorithmName;
    /** 算法类型 */
    private String type;
    /** 参数量 */
    private String params;
    /** 计算量 */
    private String flops;
    /** 算法描述 */
    private String description;
    /** 状态 */
    private int status;
    /** 去雾结果URL */
    private String resultUrl;
    /** 处理耗时(毫秒) */
    private Integer processTime;
}

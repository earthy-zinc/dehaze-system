package com.pei.dehaze.sdk.model.algorithm_select;

import lombok.Data;

import java.util.List;

/**
 * 算法详情
 * 对齐后端 AlgorithmDetailVO（/api/v1/algorithms/select/{id}）
 */
@Data
public class AlgorithmDetailVO {
    private Long id;
    private String name;
    /** 算法类型 */
    private String type;
    /** 算法图片 */
    private String img;
    private String description;
    /** 算法路径 */
    private String path;
    /** 模型文件大小 */
    private String size;
    /** 参数量 */
    private String params;
    /** FLOPs */
    private String flops;
    /** 算法版本 */
    private String version;
    /** 算法状态（0-5） */
    private Integer status;
    /** 平均评分 */
    private Double avgRating;
    /** 评价总数 */
    private Long ratingCount;
    /** 使用次数 */
    private Long usageCount;
    /** 样例效果图URL列表 */
    private List<String> sampleImages;
}

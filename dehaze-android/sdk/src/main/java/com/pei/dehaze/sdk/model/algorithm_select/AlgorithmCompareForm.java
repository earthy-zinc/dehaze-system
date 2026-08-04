package com.pei.dehaze.sdk.model.algorithm_select;

import lombok.Data;

import java.util.List;

/**
 * 算法对比表单
 * 对齐后端 AlgorithmCompareForm（/api/v1/algorithms/select/compare）
 */
@Data
public class AlgorithmCompareForm {
    /** 算法ID列表（2-3个） */
    private List<Long> algorithmIds;
    /** 文件ID（与imageUrl二选一） */
    private Long fileId;
    /** 图片URL（与fileId二选一） */
    private String imageUrl;
}

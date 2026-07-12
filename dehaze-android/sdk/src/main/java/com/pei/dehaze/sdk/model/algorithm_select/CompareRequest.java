package com.pei.dehaze.sdk.model.algorithm_select;

import java.util.List;

import lombok.Data;

/**
 * 算法对比请求
 * 对齐后端 CompareRequest
 */
@Data
public class CompareRequest {
    /** 算法ID列表（2-4个） */
    private List<Long> algorithmIds;
    /** 待对比的图片URL */
    private String imageUrl;
}

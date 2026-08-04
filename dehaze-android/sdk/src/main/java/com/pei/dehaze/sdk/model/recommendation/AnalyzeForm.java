package com.pei.dehaze.sdk.model.recommendation;

import lombok.Data;

/**
 * 图像分析请求表单
 * 对齐后端 AnalyzeForm（/api/v1/recommendations/analyze）
 */
@Data
public class AnalyzeForm {
    /** 已上传图片ID（与imageUrl二选一） */
    private Long imageId;
    /** 图片URL（与imageId二选一） */
    private String imageUrl;
}

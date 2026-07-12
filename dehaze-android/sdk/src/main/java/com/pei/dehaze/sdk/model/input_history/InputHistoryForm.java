package com.pei.dehaze.sdk.model.input_history;

import lombok.Data;

/**
 * 历史记录创建表单
 * 对齐后端 InputHistoryForm
 */
@Data
public class InputHistoryForm {
    /** 原始图片URL */
    private String originalImageUrl;
    /** 原始缩略图URL */
    private String originalThumbnailUrl;
    /** 处理结果图片URL */
    private String resultImageUrl;
    /** 结果缩略图URL */
    private String resultThumbnailUrl;
    /** 算法ID */
    private Long algorithmId;
    /** 算法名称（冗余） */
    private String algorithmName;
    /** 算法参数（JSON） */
    private String algorithmParams;
    /** 处理耗时（毫秒） */
    private Integer processingTime;
    /** 处理状态（1=成功，2=失败，3=处理中） */
    private Integer status = 3;
    /** 图片来源（upload/camera/sample） */
    private String inputSource;
    /** 是否收藏 */
    private Integer isFavorite = 0;
}

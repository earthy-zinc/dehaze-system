package com.pei.dehaze.sdk.model.input_history;

import lombok.Data;

/**
 * 历史记录更新表单（部分字段）
 * 对齐后端 InputHistoryUpdateForm
 */
@Data
public class InputHistoryUpdateForm {
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
    /** 算法名称 */
    private String algorithmName;
    /** 算法参数（JSON） */
    private String algorithmParams;
    /** 处理耗时（毫秒） */
    private Integer processingTime;
    /** 处理状态 */
    private ProcessStatus status;
    /** 是否收藏 */
    private Integer isFavorite;
    /** 同步状态（0=未同步，1=已同步） */
    private Integer syncStatus;
}

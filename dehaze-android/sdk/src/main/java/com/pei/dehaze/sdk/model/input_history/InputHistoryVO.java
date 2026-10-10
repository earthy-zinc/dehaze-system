package com.pei.dehaze.sdk.model.input_history;

import lombok.Data;

/**
 * 图像输入历史记录VO
 * 对齐后端 InputHistoryVO
 */
@Data
public class InputHistoryVO {
    /** 记录ID */
    private long id;
    /** 用户ID */
    private Long userId;
    /** 原始图片URL */
    private String originalImageUrl;
    /** 原始缩略图URL */
    private String originalThumbnailUrl;
    /** 处理结果图片URL */
    private String resultImageUrl;
    /** 结果缩略图URL */
    private String resultThumbnailUrl;
    /** 图像模型标识（统一模型规范 slug） */
    private String modelId;
    /** 模型名称 */
    private String modelName;
    /** 模型参数（JSON 字符串） */
    private String modelParams;
    /** 处理耗时（毫秒） */
    private Integer processingTime;
    /** 处理状态 */
    private ProcessStatus status;
    /** 图片来源 */
    private InputSource inputSource;
    /** 创建时间 */
    private String createTime;
    /** 更新时间 */
    private String updateTime;
}

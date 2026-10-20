package com.pei.clarity.sdk.model.input_history;

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
    /** 图像模型标识（统一模型规范 slug） */
    private String modelId;
    /** 模型名称（冗余） */
    private String modelName;
    /** 模型参数（JSON 字符串） */
    private String modelParams;
    /** 处理耗时（毫秒） */
    private Integer processingTime;
    /** 处理状态（1=成功，2=失败，3=处理中） */
    private ProcessStatus status = ProcessStatus.PROCESSING;
    /** 图片来源 */
    private InputSource inputSource;
}

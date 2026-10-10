package com.pei.dehaze.sdk.model.evaluation;

import lombok.Data;

/**
 * 评估请求参数（POST /api/v1/evaluation）
 */
@Data
public class EvalParam {
    /** 图像模型标识（统一模型规范 slug） */
    private String modelId;
    /** 预测结果文件ID（与 predUrl 至少提供一个） */
    private Long predFileId;
    /** 预测结果图片URL */
    private String predUrl;
    /** Ground Truth 参考图片文件ID（与 gtUrl 至少提供一个） */
    private Long gtFileId;
    /** Ground Truth 参考图片URL */
    private String gtUrl;
    /** 评估参数（JSON 字符串） */
    private String params;
}

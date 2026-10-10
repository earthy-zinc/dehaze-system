package com.pei.dehaze.sdk.model.image_task;

import lombok.Data;

/**
 * 图像任务响应
 * <p>
 * POST 提交返回 taskId + status（缓存命中直接 completed，含完整结果）；
 * GET 轮询按 status 返回不同字段：completed 带结果，failed 带 errorMessage。
 */
@Data
public class ImageTaskResponse {
    /** 任务ID（sys_pred_log.id） */
    private Long taskId;
    /** 任务状态（1=处理中，2=已完成，3=失败，4=已取消） */
    private ImageTaskStatus status;
    /** 预估消耗积分（提交时返回） */
    private Integer estimatedCredits;
    /** 处理后的图片URL（completed 时返回） */
    private String resultUrl;
    /** 处理结果图片MD5（completed 时返回） */
    private String resultMd5;
    /** 原始图片内容MD5（completed 时返回） */
    private String originMd5;
    /** 结果缩略图URL（completed 时返回） */
    private String resultThumbnailUrl;
    /** 处理时间（秒） */
    private Integer time;
    /** 失败错误信息（failed 时返回） */
    private String errorMessage;
}

package com.pei.dehaze.sdk.model.image_task;

import lombok.Data;

/**
 * 图像任务历史项（GET /api/v1/images/tasks）
 */
@Data
public class ImageTaskVO {
    /** 任务ID */
    private Long id;
    /** 图像模型标识（统一模型规范 slug） */
    private String model;
    /** 原图内容MD5 */
    private String originMd5;
    /** 原始图片URL */
    private String originUrl;
    /** 结果图内容MD5 */
    private String predMd5;
    /** 处理结果URL */
    private String predUrl;
    /** 任务状态（1=处理中，2=已完成，3=失败，4=已取消） */
    private ImageTaskStatus status;
    /** 失败错误信息 */
    private String errorMessage;
    /** 推理耗时（秒） */
    private Integer time;
    /** 创建时间 */
    private String createTime;
}

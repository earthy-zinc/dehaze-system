package com.pei.dehaze.sdk.model.image_task;

import com.google.gson.Gson;

import lombok.Data;

/**
 * 图像编辑任务提交表单（POST /api/v1/images/edits）
 */
@Data
public class ImageEditForm {

    /** 图像模型标识（统一模型规范 slug）；为空取平台默认模型 */
    private String model;

    /** 原始图片文件ID */
    private Long fileId;

    /** 原始图片URL（与 fileId 至少提供一个） */
    private String imageUrl;

    /** 推理参数（JSON 字符串，后端 params 字段为 String） */
    private String params;

    /** 推荐来源：推荐记录ID（推荐管理模块返回的 recommendationId） */
    private Long recommendedBy;

    /** 以去雾参数对象填充 params（序列化为 JSON 字符串） */
    public void setParams(DehazeParams dehazeParams) {
        this.params = dehazeParams == null ? null : new Gson().toJson(dehazeParams);
    }
}

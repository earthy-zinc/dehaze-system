package com.pei.clarity.sdk.model.ai_model;

import java.util.List;

import lombok.Data;

/**
 * AI 模型视图对象（GET /api/v1/ai/models/enabled）
 */
@Data
public class AiModelVO {
    /** 模型主键（sys_ai_model.id），收藏 targetId 取值 */
    private Long id;
    /** 模型业务标识（统一模型规范 slug），图像任务提交时作为 model */
    private String modelId;
    /** 模型类型（image_edit 等） */
    private String modelType;
    /** 任务归类标签（图像模型用于选择场景过滤，如 ["dehaze"]） */
    private List<String> taskTags;
    /** 显示名称 */
    private String displayName;
}

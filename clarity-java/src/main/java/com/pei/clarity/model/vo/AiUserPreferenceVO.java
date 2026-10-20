package com.pei.clarity.model.vo;

import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 用户 AI 偏好设置。
 *
 * <p>未设置偏好时 {@code defaultModel} 键不输出（对齐 python 全局剔除 null 的响应口径，
 * 不用 {@code "defaultModel": null} 表达同一语义）。
 */
@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "用户 AI 偏好设置")
public class AiUserPreferenceVO {

    @Schema(description = "偏好默认对话模型(model_id)；未设置时该键不输出，按平台默认模型解析")
    private String defaultModel;
}

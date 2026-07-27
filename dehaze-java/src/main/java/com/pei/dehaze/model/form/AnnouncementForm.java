package com.pei.dehaze.model.form;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

@Data
@Schema(description = "公告表单")
public class AnnouncementForm {

    @Schema(description = "公告标题（2-50字符）", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "公告标题不能为空")
    @Size(min = 2, max = 50, message = "公告标题长度必须在2-50个字符之间")
    private String title;

    @Schema(description = "公告内容", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "公告内容不能为空")
    private String content;

    @Schema(description = "公告类型", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "公告类型不能为空")
    private String type;

    @Schema(description = "重要级别(1:普通;2:重要)", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotNull(message = "重要级别不能为空")
    private Integer importance;

    @Schema(description = "发送范围", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "发送范围不能为空")
    private String targetScope;

    @Schema(description = "范围参数")
    private Map<String, Object> targetParams;

    @Schema(description = "定时发送时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime sendTime;

    @Schema(description = "过期时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime expireTime;
}

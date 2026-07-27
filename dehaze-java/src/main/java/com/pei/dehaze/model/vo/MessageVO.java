package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "消息视图对象")
public class MessageVO {

    @Schema(description = "消息ID")
    private Long id;

    @Schema(description = "消息类型")
    private String type;

    @Schema(description = "消息类型标签")
    private String typeLabel;

    @Schema(description = "消息标题")
    private String title;

    @Schema(description = "消息摘要（正文前50字符）")
    private String summary;

    @Schema(description = "消息正文")
    private String content;

    @Schema(description = "优先级")
    private Integer priority;

    @Schema(description = "已读状态(0:未读;1:已读)")
    private Integer readStatus;

    @Schema(description = "发送者类型(1:系统;2:管理员)")
    private Integer senderType;

    @Schema(description = "发送者类型标签")
    private String senderTypeLabel;

    @Schema(description = "已读时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime readTime;

    @Schema(description = "跳转链接")
    private String jumpUrl;

    @Schema(description = "扩展数据")
    private Map<String, Object> extra;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}

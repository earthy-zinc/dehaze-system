package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Data
@Schema(description = "反馈列表VO")
public class FeedbackPageVO {

    @Schema(description = "反馈ID")
    private Long id;

    @Schema(description = "用户ID")
    private Long userId;

    @Schema(description = "用户名")
    private String username;

    @Schema(description = "反馈类型")
    private String feedbackType;

    @Schema(description = "反馈标题")
    private String title;

    @Schema(description = "反馈内容")
    private String content;

    @Schema(description = "反馈状态")
    private String status;

    @Schema(description = "优先级")
    private Integer priority;

    @Schema(description = "处理人ID")
    private Long assigneeId;

    @Schema(description = "处理人名称")
    private String assigneeName;

    @Schema(description = "相关模块")
    private String relatedModule;

    @Schema(description = "反馈标签")
    private List<String> tags;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "更新时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;
}

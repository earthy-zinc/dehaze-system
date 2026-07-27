package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "公告视图对象")
public class AnnouncementVO {

    @Schema(description = "公告ID")
    private Long id;

    @Schema(description = "公告标题")
    private String title;

    @Schema(description = "公告内容")
    private String content;

    @Schema(description = "公告类型")
    private String type;

    @Schema(description = "公告类型标签")
    private String typeLabel;

    @Schema(description = "重要级别(1:普通;2:重要)")
    private Integer importance;

    @Schema(description = "重要级别标签")
    private String importanceLabel;

    @Schema(description = "发送范围")
    private String targetScope;

    @Schema(description = "发送范围标签")
    private String targetScopeLabel;

    @Schema(description = "公告状态(1:草稿;2:待发送;3:已发送;4:已取消)")
    private Integer status;

    @Schema(description = "公告状态标签")
    private String statusLabel;

    @Schema(description = "发送时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime sendTime;

    @Schema(description = "过期时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime expireTime;

    @Schema(description = "已发送人数")
    private Integer sentCount;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "创建人ID")
    private Long createBy;
}

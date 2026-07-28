package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;
import java.util.List;

@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "反馈详情VO")
public class FeedbackDetailVO extends FeedbackPageVO {

    @Schema(description = "联系方式")
    private String contact;

    @Schema(description = "截图URL")
    private List<String> images;

    @Schema(description = "分配时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime assignedTime;

    @Schema(description = "关闭原因")
    private String closeReason;

    @Schema(description = "回复列表")
    private List<FeedbackReplyVO> replies;
}

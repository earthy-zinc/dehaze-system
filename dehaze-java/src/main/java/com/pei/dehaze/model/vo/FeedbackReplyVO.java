package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Data
@Schema(description = "反馈回复VO")
public class FeedbackReplyVO {

    @Schema(description = "回复ID")
    private Long id;

    @Schema(description = "反馈ID")
    private Long feedbackId;

    @Schema(description = "回复人ID")
    private Long replierId;

    @Schema(description = "回复人名称")
    private String replierName;

    @Schema(description = "回复人类型(1:用户;2:管理员)")
    private Integer replierType;

    @Schema(description = "回复内容")
    private String content;

    @Schema(description = "回复类型(info/resolved/unsupported/dev_transfer)")
    private String replyType;

    @Schema(description = "附件URL")
    private List<String> attachments;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}

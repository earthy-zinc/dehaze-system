package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/** 计费明细记录 */
@Schema(description = "计费明细记录")
@Data
public class AiBillingRecordVO {

    private Long id;

    private Long userId;

    private Long conversationId;

    private Long messageId;

    private String model;

    /** 用户原选模型标识（NULL 表示未降级） */
    private String actualModel;

    private String billType;

    private Integer inputTokens;

    private Integer cachedInputTokens;

    private Integer outputTokens;

    private Integer credits;

    private Integer creditsSaved;

    private Integer toolCredits;

    private Integer quotaConsumed;

    private Integer preDeduct;

    /** 误扣申诉状态(0:无;1:待审核;2:已通过;3:已驳回) */
    private Integer refundStatus = 0;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}

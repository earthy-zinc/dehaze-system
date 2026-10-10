package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.fasterxml.jackson.databind.ser.std.ToStringSerializer;
import io.swagger.v3.oas.annotations.media.Schema;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import lombok.Data;

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

    /** 非 token 计量数量（张/秒/万字符；token 类模型为 null）；python Decimal 序列化为字符串，java 对齐同一口径 */
    @JsonSerialize(using = ToStringSerializer.class)
    private BigDecimal quantity;

    /** 非 token 计量单位（image:张;second:秒;ten_thousand_chars:万字符;frame:帧） */
    private String quantityUnit;

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

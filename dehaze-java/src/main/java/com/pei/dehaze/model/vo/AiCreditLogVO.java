package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.fasterxml.jackson.databind.ser.std.ToStringSerializer;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/** 积分余额变动流水 */
@Schema(description = "积分余额变动流水")
@Data
public class AiCreditLogVO {

    private Long id;

    private Long userId;

    private String source;

    /** 金额按 python Decimal 序列化口径下发为字符串，避免跨端精度表示不一致 */
    @JsonSerialize(using = ToStringSerializer.class)
    private Long amount;

    @JsonSerialize(using = ToStringSerializer.class)
    private Long balanceAfter;

    private Long relatedId;

    private String reason;

    private Long operatorId;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}

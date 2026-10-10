package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.fasterxml.jackson.databind.ser.std.ToStringSerializer;
import io.swagger.v3.oas.annotations.media.Schema;
import java.math.BigDecimal;
import lombok.Data;

/** 用户余额账户视图（权益缺失/停用时限额展示为 0） */
@Schema(description = "用户余额账户视图")
@Data
public class AiBalanceVO {

    private Long userId;

    /** 积分余额：python `BalanceResult.credits_balance` 为 Decimal → JSON 字符串，java 必须同口径 */
    @JsonSerialize(using = ToStringSerializer.class)
    private BigDecimal creditsBalance;

    private Boolean arrearsStatus;

    private Integer dailyUsed;

    private Integer dailyLimit;

    private Integer monthlyUsed;

    private Integer monthlyLimit;
}

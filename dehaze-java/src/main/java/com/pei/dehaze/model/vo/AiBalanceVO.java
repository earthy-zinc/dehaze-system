package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.math.BigDecimal;

/** 用户余额账户视图（权益缺失/停用时限额展示为 0） */
@Schema(description = "用户余额账户视图")
@Data
public class AiBalanceVO {

    private Long userId;

    private BigDecimal creditsBalance;

    private Boolean arrearsStatus;

    private Integer dailyUsed;

    private Integer dailyLimit;

    private Integer monthlyUsed;

    private Integer monthlyLimit;
}

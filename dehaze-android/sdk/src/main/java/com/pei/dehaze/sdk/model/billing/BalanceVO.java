package com.pei.dehaze.sdk.model.billing;

import lombok.Data;

/**
 * 用户余额与限额概况（GET /api/v1/ai-billing/balance）
 */
@Data
public class BalanceVO {
    /** 用户ID */
    private Long userId;
    /** 积分余额（后端 Decimal 防精度丢失按字符串下发，如 "100.00"） */
    private String creditsBalance;
    /** 是否欠费 */
    private Boolean arrearsStatus;
    /** 今日已用积分 */
    private Integer dailyUsed;
    /** 日积分限额 */
    private Integer dailyLimit;
    /** 本月已用积分 */
    private Integer monthlyUsed;
    /** 月积分限额 */
    private Integer monthlyLimit;
}

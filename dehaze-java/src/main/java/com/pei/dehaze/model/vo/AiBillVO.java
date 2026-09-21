package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.math.BigDecimal;
import java.util.Map;

/** 月结账单 */
@Schema(description = "月结账单")
@Data
public class AiBillVO {

    private Long userId;

    /** 账期月份（YYYY-MM） */
    private String month;

    private Integer totalConsume;

    private Integer totalRecharge;

    private Integer totalRefund;

    private BigDecimal balanceStart;

    private BigDecimal balanceEnd;

    /** 按 billType 维度的明细汇总 */
    private Map<String, Integer> itemSummary;
}

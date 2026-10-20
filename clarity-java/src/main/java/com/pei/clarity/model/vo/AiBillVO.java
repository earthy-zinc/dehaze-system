package com.pei.clarity.model.vo;

import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.fasterxml.jackson.databind.ser.std.ToStringSerializer;
import io.swagger.v3.oas.annotations.media.Schema;
import java.math.BigDecimal;
import java.util.Map;
import lombok.Data;

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

    /** 期初余额（python Decimal 序列化为字符串，java 对齐同一口径；该 VO 亦为 `ai:bill:*` 缓存载体） */
    @JsonSerialize(using = ToStringSerializer.class)
    private BigDecimal balanceStart;

    /** 期末余额（同上，按字符串下发） */
    @JsonSerialize(using = ToStringSerializer.class)
    private BigDecimal balanceEnd;

    /** 按 billType 维度的明细汇总 */
    private Map<String, Integer> itemSummary;
}

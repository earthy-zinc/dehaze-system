package com.pei.dehaze.model.read;

import lombok.Data;

import java.math.BigDecimal;

/** 积分流水按来源汇总行 */
@Data
public class AiCreditSourceRead {

    private String source;

    private BigDecimal amount;
}

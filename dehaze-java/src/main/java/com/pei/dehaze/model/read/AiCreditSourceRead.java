package com.pei.dehaze.model.read;

import java.math.BigDecimal;
import lombok.Data;

/** 积分流水按来源汇总行 */
@Data
public class AiCreditSourceRead {

    private String source;

    private BigDecimal amount;
}

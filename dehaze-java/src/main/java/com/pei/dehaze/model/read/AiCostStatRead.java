package com.pei.dehaze.model.read;

import java.math.BigDecimal;
import lombok.Data;

/** 成本聚合行（按 model/provider 维度） */
@Data
public class AiCostStatRead {

    private String dimension;

    private BigDecimal cost;
}

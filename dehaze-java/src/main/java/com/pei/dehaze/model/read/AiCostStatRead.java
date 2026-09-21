package com.pei.dehaze.model.read;

import lombok.Data;

import java.math.BigDecimal;

/** 成本聚合行（按 model/provider 维度） */
@Data
public class AiCostStatRead {

    private String dimension;

    private BigDecimal cost;
}

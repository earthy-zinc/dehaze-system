package com.pei.dehaze.model.read;

import lombok.Data;

/** 已实收订单按商品类型汇总行（分） */
@Data
public class AiOrderIncomeRead {

    private String packageType;

    private Long amount;
}

package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 成本-利润统计（毛利核算双口径）。
 *
 * <p>分组维度（model/provider）仅输出成本分解——订单实收无法按模型/供应商归因，
 * 分组行不携带收入/毛利/口径字段。
 */
@Schema(description = "成本-利润统计")
@Data
public class AiCostStatVO {

    private String dimension;

    private Double revenue;

    private Double cost;

    private Double profit;

    private Double profitRate;

    /** overall:整体毛利官方口径;ai:AI 参考口径（分组维度不返回） */
    private String metric;
}

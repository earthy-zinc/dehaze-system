package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
@Schema(description = "订单统计VO")
public class OrderStatsVO {

    @Schema(description = "总订单数")
    private Long totalOrders;

    @Schema(description = "总收入（分）")
    private Long totalRevenue;

    @Schema(description = "总退款金额（分）")
    private Long totalRefund;

    @Schema(description = "退款率")
    private Double refundRate;

    @Schema(description = "订单状态分布")
    private Map<String, Long> statusDistribution;

    @Schema(description = "支付方式分布")
    private Map<String, Long> payMethodDistribution;

    @Schema(description = "套餐销售分布")
    private List<PackageStatItem> packageDistribution;

    @Schema(description = "每日统计")
    private List<DailyStatItem> dailyStats;

    @Data
    @Schema(description = "套餐销售统计项")
    public static class PackageStatItem {
        private Long packageId;
        private String packageName;
        private Long count;
        private Long revenue;
    }

    @Data
    @Schema(description = "每日统计项")
    public static class DailyStatItem {
        private String date;
        private Long count;
        private Long revenue;
    }
}

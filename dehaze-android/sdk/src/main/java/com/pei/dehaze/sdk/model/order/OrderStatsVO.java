package com.pei.dehaze.sdk.model.order;

import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
public class OrderStatsVO {
    private Integer totalOrders;
    private Double totalRevenue;
    private Double totalRefund;
    private Double refundRate;
    private Map<String, Integer> statusDistribution;
    private Map<String, Integer> payMethodDistribution;
    private List<DistItem> packageDistribution;
    private List<DailyStat> dailyStats;

    @Data
    public static class DistItem {
        private Long packageId;
        private String packageName;
        private Integer count;
        private Double revenue;
    }

    @Data
    public static class DailyStat {
        private String date;
        private Integer count;
        private Double revenue;
    }
}

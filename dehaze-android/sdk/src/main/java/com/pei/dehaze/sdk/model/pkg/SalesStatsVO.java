package com.pei.dehaze.sdk.model.pkg;

import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
public class SalesStatsVO {
    private Integer totalSales;
    private Double totalRevenue;
    private List<StatItem> packageStats;
    private List<StatItem> levelStats;
    private List<StatItem> periodStats;
    private CouponStats couponStats;

    @Data
    public static class StatItem {
        private Long packageId;
        private String packageName;
        private String levelCode;
        private String levelName;
        private String period;
        private String periodName;
        private Integer salesCount;
        private Integer count;
        private Double revenue;
    }

    @Data
    public static class CouponStats {
        private Integer totalIssued;
        private Integer totalUsed;
        private Double usageRate;
    }
}

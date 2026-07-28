package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;

@Data
@Schema(description = "销售统计VO")
public class SalesStatsVO {

    @Schema(description = "总销量")
    private Long totalSales;

    @Schema(description = "总收入（分）")
    private Long totalRevenue;

    @Schema(description = "套餐销售统计")
    private List<PackageStatItem> packageStats;

    @Schema(description = "等级销售统计")
    private List<LevelStatItem> levelStats;

    @Schema(description = "周期销售统计")
    private List<PeriodStatItem> periodStats;

    @Schema(description = "优惠券统计")
    private CouponStatItem couponStats;

    @Data
    @Schema(description = "套餐销售统计项")
    public static class PackageStatItem {
        private Long packageId;
        private String packageName;
        private Long salesCount;
        private Long revenue;
    }

    @Data
    @Schema(description = "等级销售统计项")
    public static class LevelStatItem {
        private String levelCode;
        private String levelName;
        private Long salesCount;
        private Long revenue;
    }

    @Data
    @Schema(description = "周期销售统计项")
    public static class PeriodStatItem {
        private String period;
        private String periodName;
        private Long salesCount;
        private Long revenue;
    }

    @Data
    @Schema(description = "优惠券统计项")
    public static class CouponStatItem {
        private Long totalIssued;
        private Long totalUsed;
        private Double usageRate;
    }
}

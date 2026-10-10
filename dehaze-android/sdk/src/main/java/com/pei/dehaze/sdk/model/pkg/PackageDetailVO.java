package com.pei.dehaze.sdk.model.pkg;

import lombok.Data;

import java.util.Map;

@Data
public class PackageDetailVO {
    private Long id;
    private String name;
    private String levelCode;
    private String levelName;
    private String period;
    private Integer periodDays;
    private Long originalPrice;
    private Long salePrice;
    private Long dailyPrice;
    private String description;
    private Map<String, Integer> benefits;
    private Object[] activePromotions;
    private Integer salesCount;
}

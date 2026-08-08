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
    private Double originalPrice;
    private Double salePrice;
    private Double dailyPrice;
    private String description;
    private Map<String, Integer> benefits;
    private Object[] activePromotions;
    private Integer salesCount;
}

package com.pei.dehaze.sdk.model.pkg;

import lombok.Data;

@Data
public class PackagePageVO {
    private Long id;
    private String name;
    private String levelCode;
    private String levelName;
    private String period;
    private Integer periodDays;
    private Double originalPrice;
    private Double salePrice;
    private Double dailyPrice;
    private Integer salesCount;
    private Integer status;
    private String createTime;
}

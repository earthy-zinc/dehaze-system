package com.pei.clarity.sdk.model.pkg;

import lombok.Data;

@Data
public class PackagePageVO {
    private Long id;
    private String name;
    private String levelCode;
    private String levelName;
    private String period;
    private Integer periodDays;
    private Long originalPrice;
    private Long salePrice;
    private Long dailyPrice;
    private Integer salesCount;
    private Integer status;
    private String createTime;
}

package com.pei.dehaze.sdk.model.pkg;

import lombok.Data;

@Data
public class PackageForm {
    private Long id;
    private String name;
    private String levelCode;
    private String period;
    private Integer periodDays;
    private Double originalPrice;
    private Double salePrice;
    private String description;
    private BenefitOverrides benefitOverrides;
    private Integer sort;
    private Integer status;
}

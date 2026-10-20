package com.pei.clarity.sdk.model.pkg;

import lombok.Data;

@Data
public class PackageForm {
    private Long id;
    private String name;
    private String levelCode;
    private String period;
    private Integer periodDays;
    private Long originalPrice;
    private Long salePrice;
    private String description;
    private BenefitOverrides benefitOverrides;
    private Integer sort;
    private Integer status;
}

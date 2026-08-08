package com.pei.dehaze.sdk.model.prediction;

import lombok.Data;

@Data
public class PredictionQuota {
    private Integer remaining;
    private Integer total;
    private Integer used;
    private String resetDate;
}

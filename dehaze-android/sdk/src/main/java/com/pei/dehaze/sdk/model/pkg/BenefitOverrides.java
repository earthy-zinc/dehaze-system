package com.pei.dehaze.sdk.model.pkg;

import lombok.Data;

@Data
public class BenefitOverrides {
    private Integer monthlyDehazeQuota;
    private Integer monthlyEvaluateQuota;
    private Integer historyRetention;
    private Integer batchLimit;
    private Integer priority;
    private Integer advancedParams;
    private Integer hdExport;
    private Integer reportExport;
    private Integer batchDownload;
}

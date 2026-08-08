package com.pei.dehaze.sdk.model.member;

import lombok.Data;

@Data
public class BenefitVO {
    private String levelCode;
    private String levelName;
    private Integer growthMin;
    private Integer growthMax;
    private Integer monthlyDehazeQuota;
    private Integer monthlyEvaluateQuota;
    private Integer historyRetention;
    private Integer batchLimit;
    private Integer priority;
    private Integer advancedParams;
    private Integer hdExport;
    private Integer reportExport;
    private Integer batchDownload;
    private Integer sort;
    private Integer status;
}

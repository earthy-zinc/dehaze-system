package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "算法对比结果项")
public class AlgorithmCompareVO {

    @Schema(description = "算法ID")
    private Long algorithmId;

    @Schema(description = "算法名称")
    private String algorithmName;

    @Schema(description = "处理结果URL")
    private String resultUrl;

    @Schema(description = "处理耗时（毫秒）")
    private Integer time;

    @Schema(description = "评估指标（PSNR/SSIM等，JSON字符串）")
    private String metrics;
}

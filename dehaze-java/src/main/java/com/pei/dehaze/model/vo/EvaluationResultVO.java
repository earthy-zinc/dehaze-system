package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.Map;

@Schema(description = "评估结果视图对象")
@Data
public class EvaluationResultVO {

    @Schema(description = "评估日志ID")
    private Long logId;

    @Schema(description = "任务状态：processing/completed/failed")
    private String status;

    @Schema(description = "评估指标结果（PSNR/SSIM/LPIPS/NIQE/等，status=completed 时返回）")
    private Map<String, Double> metrics;

    @Schema(description = "处理时间（毫秒）")
    private Integer time;

    @Schema(description = "失败错误信息（status=failed 时返回）")
    private String errorMessage;
}

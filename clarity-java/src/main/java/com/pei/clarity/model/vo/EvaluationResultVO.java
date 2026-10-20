package com.pei.clarity.model.vo;

import com.pei.clarity.common.enums.LogStatusEnum;
import io.swagger.v3.oas.annotations.media.Schema;
import java.util.Map;
import lombok.Data;

@Schema(description = "评估结果视图对象")
@Data
public class EvaluationResultVO {

    @Schema(description = "评估日志ID")
    private Long logId;

    @Schema(description = "任务状态：1=处理中,2=已完成,3=失败")
    private LogStatusEnum status;

    @Schema(description = "评估指标结果（PSNR/SSIM/LPIPS/NIQE/等，status=completed 时返回）")
    private Map<String, Double> metrics;

    @Schema(description = "预测图内容MD5（status=completed 时返回）")
    private String predMd5;

    @Schema(description = "参考图内容MD5（status=completed 时返回）")
    private String gtMd5;

    @Schema(description = "处理时间（秒）")
    private Integer time;

    @Schema(description = "失败错误信息（status=failed 时返回）")
    private String errorMessage;
}

package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Schema(description = "预测结果视图对象")
@Data
public class PredictionResultVO {

    @Schema(description = "预测日志ID")
    private Long logId;

    @Schema(description = "任务状态：processing/completed/failed")
    private String status;

    @Schema(description = "处理后的图片URL（status=completed 时返回）")
    private String resultUrl;

    @Schema(description = "处理后的缩略图URL（status=completed 时返回）")
    private String resultThumbnailUrl;

    @Schema(description = "处理时间（毫秒）")
    private Integer time;

    @Schema(description = "失败错误信息（status=failed 时返回）")
    private String errorMessage;
}

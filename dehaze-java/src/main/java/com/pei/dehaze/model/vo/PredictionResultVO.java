package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Schema(description = "图像任务响应视图对象")
@Data
public class PredictionResultVO {

    @Schema(description = "任务ID（sys_pred_log.id）")
    private Long taskId;

    @Schema(description = "任务状态：1=处理中,2=已完成,3=失败,4=已取消")
    private Integer status;

    @Schema(description = "预估消耗积分（提交时返回）")
    private Integer estimatedCredits;

    @Schema(description = "处理后的图片URL（status=completed 时返回）")
    private String resultUrl;

    @Schema(description = "处理结果图片MD5（status=completed 时返回）")
    private String resultMd5;

    @Schema(description = "原始图片内容MD5（status=completed 时返回）")
    private String originMd5;

    @Schema(description = "处理后的缩略图URL（status=completed 时返回）")
    private String resultThumbnailUrl;

    @Schema(description = "处理时间（秒）")
    private Integer time;

    @Schema(description = "失败错误信息（status=failed 时返回）")
    private String errorMessage;
}

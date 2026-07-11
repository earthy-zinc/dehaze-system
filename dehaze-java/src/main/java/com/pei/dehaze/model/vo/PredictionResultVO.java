package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 预测结果视图对象
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Schema(description = "预测结果视图对象")
@Data
public class PredictionResultVO {

    @Schema(description = "预测日志ID")
    private Long logId;

    @Schema(description = "处理后的图片URL")
    private String resultUrl;

    @Schema(description = "处理后的缩略图URL")
    private String resultThumbnailUrl;

    @Schema(description = "处理时间（毫秒）")
    private Integer time;
}

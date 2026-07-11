package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 历史记录创建表单
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Schema(description = "历史记录创建表单")
@Data
public class HistoryForm {

    @Schema(description = "原始图片URL")
    private String originalImageUrl;

    @Schema(description = "原始缩略图URL")
    private String originalThumbnailUrl;

    @Schema(description = "处理结果图片URL")
    private String resultImageUrl;

    @Schema(description = "结果缩略图URL")
    private String resultThumbnailUrl;

    @Schema(description = "算法ID")
    private Long algorithmId;

    @Schema(description = "算法名称")
    private String algorithmName;

    @Schema(description = "算法参数（JSON）")
    private String algorithmParams;

    @Schema(description = "处理耗时（毫秒）")
    private Integer processingTime;

    @Schema(description = "处理状态（1=成功，2=失败，3=处理中）")
    private Integer status;

    @Schema(description = "图片来源（upload/camera/sample）")
    private String inputSource;
}

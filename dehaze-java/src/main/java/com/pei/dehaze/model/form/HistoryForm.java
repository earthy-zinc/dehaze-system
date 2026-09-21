package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Data;

/**
 * 历史记录创建表单
 *
 * <p>长度约束与 sys_input_history 列定义一致，超长入库会导致 MySQL 报错。
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Schema(description = "历史记录创建表单")
@Data
public class HistoryForm {

    @Schema(description = "原始图片URL")
    @Size(max = 500, message = "原始图片URL长度不能超过500")
    private String originalImageUrl;

    @Schema(description = "原始缩略图URL")
    @Size(max = 500, message = "原始缩略图URL长度不能超过500")
    private String originalThumbnailUrl;

    @Schema(description = "处理结果图片URL")
    @Size(max = 500, message = "处理结果图片URL长度不能超过500")
    private String resultImageUrl;

    @Schema(description = "结果缩略图URL")
    @Size(max = 500, message = "结果缩略图URL长度不能超过500")
    private String resultThumbnailUrl;

    @Schema(description = "算法ID")
    private Long algorithmId;

    @Schema(description = "算法名称")
    @Size(max = 100, message = "算法名称长度不能超过100")
    private String algorithmName;

    @Schema(description = "算法参数（JSON）")
    private String algorithmParams;

    @Schema(description = "处理耗时（毫秒）")
    @Min(value = 0, message = "处理耗时不能为负数")
    private Integer processingTime;

    @Schema(description = "处理状态（1=成功，2=失败，3=处理中）")
    @Min(value = 1, message = "处理状态取值无效")
    @Max(value = 3, message = "处理状态取值无效")
    private Integer status;

    @Schema(description = "图片来源（upload/camera/sample）")
    @Pattern(regexp = "upload|camera|sample", message = "图片来源取值无效")
    private String inputSource;
}

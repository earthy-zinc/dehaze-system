package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 批量上传成功项VO
 *
 * @author earthy-zinc
 * @since 2025-01-10
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "批量上传成功项视图对象")
public class BatchUploadSuccessItemVO {

    @Schema(description = "数据项ID", example = "1")
    private Long id;

    @Schema(description = "数据项名称", example = "street_001")
    private String name;

    @Schema(description = "文件数量（清晰图+有雾图）", example = "3")
    private Integer fileCount;
}

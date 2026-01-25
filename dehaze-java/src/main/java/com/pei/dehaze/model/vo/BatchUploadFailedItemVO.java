package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 批量上传失败项VO
 *
 * @author earthy-zinc
 * @since 2025-01-10
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "批量上传失败项视图对象")
public class BatchUploadFailedItemVO {

    @Schema(description = "文件名", example = "invalid.jpg")
    private String fileName;

    @Schema(description = "失败原因", example = "未找到配对的清晰图")
    private String reason;
}

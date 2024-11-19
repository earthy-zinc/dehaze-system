package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Schema(description = "图片表单对象")
@Data
public class ImageForm {
    @Schema(description = "所属数据集id")
    private Long datasetId;

    @Schema(description = "数据项id")
    private Long imageItemId;

    @Schema(description = "图片类型（清晰图、雾霾图、分割图等）")
    private String type;
}

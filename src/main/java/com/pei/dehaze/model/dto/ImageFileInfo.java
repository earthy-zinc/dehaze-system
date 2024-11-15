package com.pei.dehaze.model.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ImageFileInfo {

    @Schema(description = "当前图片id")
    private Long id;

    @Schema(description = "所属数据集id")
    private Long datasetId;

    @Schema(description = "所属数据项id")
    private Long imageItemId;

    @Schema(description = "所属文件id")
    private Long fileId;

    @Schema(description = "图片类型")
    private String type;

    @Schema(description = "文件名称")
    private String name;

    @Schema(description = "文件URL")
    private String url;
}

package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:16:10
 */
@Schema(description = "算法视图对象")
@Data
public class AlgorithmVO {
    @Schema(description = "算法ID")
    private long id;

    @Schema(description = "算法名称")
    private String name;

    @Schema(description = "算法类型")
    private String type;

    @Schema(description = "算法图片")
    private String img;

    @Schema(description = "算法描述")
    private String description;

    @Schema(description = "算法路径")
    private String path;

    @Schema(description = "算法浮点数")
    private String flops;

    @Schema(description = "算法参数量")
    private String params;

    @Schema(description = "导入路径")
    private String importPath;

    @Schema(description = "开启关闭状态")
    private Integer status;

    @Schema(description = "算法大小")
    private String size;

    @Schema(description = "子算法")
    private List<AlgorithmVO> children;
}

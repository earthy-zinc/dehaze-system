package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;

@Data
@Schema(description = "算法详情视图对象（含样例效果图、评分、使用次数）")
public class AlgorithmDetailVO {

    @Schema(description = "算法ID")
    private Long id;

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

    @Schema(description = "模型文件大小")
    private String size;

    @Schema(description = "参数量")
    private String params;

    @Schema(description = "FLOPs")
    private String flops;

    @Schema(description = "算法版本")
    private String version;

    @Schema(description = "算法状态")
    private Integer status;

    @Schema(description = "平均评分")
    private Double avgRating;

    @Schema(description = "评价总数")
    private Long ratingCount;

    @Schema(description = "使用次数")
    private Long usageCount;

    @Schema(description = "样例效果图URL列表（从数据集样例获取）")
    private List<String> sampleImages;
}

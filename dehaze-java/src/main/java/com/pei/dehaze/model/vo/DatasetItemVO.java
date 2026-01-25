package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 数据项详情VO
 *
 * @author earthy-zinc
 * @since 2024-12-07
 */
@Data
@Schema(description = "数据项详情视图对象")
public class DatasetItemVO {

    @Schema(description = "数据项ID", example = "1")
    private Long id;

    @Schema(description = "所属数据集ID", example = "1")
    private Long datasetId;

    @Schema(description = "数据项名称，用于标识该数据项", example = "城市街道_001")
    private String name;

    @Schema(description = "场景类型，用户自定义的场景分类（如：城市街道、山区风景、海边等）", example = "城市街道")
    private String sceneType;

    @Schema(description = "数据项描述信息", example = "城市主干道雾霾场景")
    private String description;

    @Schema(description = "使用次数，记录该数据项被使用的次数", example = "15")
    private Integer usageCount;

    @Schema(description = "该数据项包含的图片总数（清晰图+有雾图）", example = "3")
    private Integer imageCount;

    @Schema(description = "清晰图信息（Ground Truth），每个数据项只有一张清晰图")
    private ImageUrlVO clearImage;

    @Schema(description = "有雾图列表，每个数据项可以有多张不同雾霾程度的有雾图")
    private List<ImageUrlVO> hazyImages;

    @Schema(description = "数据项创建时间", example = "2024-01-01 10:00:00")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "数据项最后更新时间", example = "2024-01-15 15:30:00")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;
}

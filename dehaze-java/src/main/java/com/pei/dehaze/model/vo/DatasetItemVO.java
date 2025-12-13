package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.pei.dehaze.model.bo.ItemFileBO;
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

    @Schema(description = "数据项ID")
    private Long id;

    @Schema(description = "数据集ID")
    private Long datasetId;

    @Schema(description = "数据项名称")
    private String name;

    @Schema(description = "场景类型")
    private String sceneType;

    @Schema(description = "图片总数")
    private Integer imageCount;

    @Schema(description = "清晰图信息")
    private ImageUrlVO clearImage;

    @Schema(description = "有雾图列表")
    private List<ImageUrlVO> hazyImages;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "更新时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;
}

package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 图片详情VO
 *
 * @author earthy-zinc
 * @since 2025-12-07
 */
@Data
@Schema(description = "图片详情")
public class ImageDetailVO {

    @Schema(description = "图片ID")
    private Long id;

    @Schema(description = "数据项ID")
    private Long itemId;

    @Schema(description = "数据集ID")
    private Long datasetId;

    @Schema(description = "数据集名称")
    private String datasetName;

    @Schema(description = "文件名")
    private String fileName;

    @Schema(description = "文件大小")
    private String fileSize;

    @Schema(description = "文件格式")
    private String fileFormat;

    @Schema(description = "宽度")
    private Integer width;

    @Schema(description = "高度")
    private Integer height;

    @Schema(description = "分辨率")
    private String resolution;

    @Schema(description = "图片类型")
    private String type;

    @Schema(description = "描述")
    private String description;

    @Schema(description = "场景类型")
    private String sceneType;

    @Schema(description = "雾霾程度")
    private String hazeLevel;

    @Schema(description = "使用次数")
    private Long usageCount;

    @Schema(description = "原图URL")
    private String url;

    @Schema(description = "缩略图URL")
    private String thumbnailUrl;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "是否有配对图片")
    private Boolean hasPairedImages;

    @Schema(description = "配对图片数量")
    private Integer pairedCount;
}

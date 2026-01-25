package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 图片简要视图对象（用于列表展示和配对关系展示）
 * 不包含循环引用的pairedFiles和datasetItem字段
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@Data
@Schema(description = "图片简要视图对象")
public class SimpleImageUrlVO {
    @Schema(description = "数据项文件ID", example = "1")
    private Long id;

    @Schema(description = "所属数据项ID", example = "1")
    private Long itemId;

    @Schema(description = "所属数据集ID", example = "1")
    private Long datasetId;

    @Schema(description = "图片类型：clear-清晰图，hazy-有雾图", example = "hazy")
    private String type;

    @Schema(description = "图片访问URL（经过处理的URL）", example = "http://example.com/images/sample.jpg")
    private String url;

    @Schema(description = "缩略图URL，用于列表展示", example = "http://example.com/thumbnails/sample_thumb.jpg")
    private String thumbnailUrl;

    @Schema(description = "图片描述信息", example = "城市街道场景的有雾图像")
    private String description;

    @Schema(description = "图片宽度（像素）", example = "1920")
    private Integer width;

    @Schema(description = "图片高度（像素）", example = "1080")
    private Integer height;

    @Schema(description = "雾霾程度：light-轻度，medium-中度，heavy-重度", example = "medium")
    private String hazeLevel;

    @Schema(description = "文件名", example = "scene_001_hazy.jpg")
    private String fileName;

    @Schema(description = "文件大小（字节）", example = "2560000")
    private Long sizeBytes;

    @Schema(description = "文件大小，格式化显示", example = "2.44MB")
    private String formattedSize;

    @Schema(description = "文件格式", example = "jpg", allowableValues = {"jpg", "jpeg", "webp", "png", "gif"})
    private String format;

    @Schema(description = "图片创建时间", example = "2024-01-01 10:00:00")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}

package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonIgnore;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:47:10
 */
@Data
@Schema(description = "图片详情视图对象")
public class ImageUrlVO {
    @Schema(description = "数据项文件ID", example = "1")
    private Long id;

    @Schema(description = "所属数据项ID", example = "1")
    private Long itemId;

    @Schema(description = "所属数据集ID", example = "1")
    private Long datasetId;

    @Schema(description = "所属数据集名称", example = "测试数据集")
    private String datasetName;

    @Schema(description = "所属数据项信息", example = "")
    private DatasetItemSimpleVO datasetItem;

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

    @Schema(description = "场景类型", example = "城市街道")
    private String sceneType;

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

    @Schema(description = "文件MD5值", example = "abc123...")
    private String md5;

    @Schema(description = "使用次数，记录该图片被用于去雾处理的次数", example = "10")
    private Long usageCount;

    @Schema(description = "图片上传时间", example = "2024-01-01 10:00:00")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "是否有配对图片，true表示该图片属于配对组", example = "true")
    private Boolean hasPairedImages;

    @Schema(description = "配对图片列表（属于同一数据项的其他图片）", example = "")
    private List<SimpleImageUrlVO> pairedFiles;

    @Schema(description = "配对图片总数（包括清晰图和所有有雾图）", example = "3")
    private Integer pairedCount;

    // 以下字段不序列化，仅用于 mapper 映射 objectName + storage，由 service 层拼接 url
    @JsonIgnore
    private String objectName;
    @JsonIgnore
    private String storage;
    @JsonIgnore
    private String thumbnailObjectName;
    @JsonIgnore
    private String thumbnailStorage;
}

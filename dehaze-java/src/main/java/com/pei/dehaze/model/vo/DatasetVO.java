package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:20:13
 */
@Schema(description = "数据集视图对象")
@Data
public class DatasetVO {
    @Schema(description = "数据集ID")
    private Long id;

    @Schema(description = "父数据集ID")
    private Long parentId;

    @Schema(description = "数据集类型")
    private String type;

    @Schema(description = "数据集名称")
    private String name;

    @Schema(description = "数据集描述")
    private String description;

    @Schema(description = "存储位置")
    private String path;

    @Schema(description = "占用空间大小")
    private String size;

    @Schema(description = "样例图片URL")
    private String img;

    @Schema(description = "图片数量")
    private Long imageCount;

    @Schema(description = "使用次数")
    private Long usageCount;

    @Schema(description = "子数据集")
    private List<DatasetVO> children;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm")
    private LocalDateTime createTime;

    @Schema(description = "修改时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm")
    private LocalDateTime updateTime;

    @Schema(description = "状态(1:启用；0:禁用)")
    private Integer status;

    @Schema(description = "场景分布")
    private Map<String, Long> sceneDistribution;

    @Schema(description = "雾霾程度分布")
    private Map<String, Long> hazeDistribution;

    @Schema(description = "文件格式分布")
    private Map<String, Long> formatDistribution;
}

package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.pei.dehaze.model.dto.DatasetStatistics;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:20:13
 */
@Schema(description = "数据集视图对象")
@Data
public class DatasetVO {
    @Schema(description = "数据集ID", example = "1")
    private Long id;

    @Schema(description = "父数据集ID，null表示根数据集", example = "null")
    private Long parentId;

    @Schema(description = "数据集类型：training, test, user, result", example = "training")
    private String type;

    @Schema(description = "数据集名称", example = "我的测试数据集")
    private String name;

    @Schema(description = "数据集描述信息", example = "用于测试去雾算法的数据集")
    private String description;

    @Schema(description = "数据集存储路径", example = "/户外场景数据集")
    private String path;

    @Schema(description = "是否有子数据集", example = "true")
    private Boolean hasChildren;

    @Schema(description = "子数据集列表，支持多级嵌套树形结构")
    private List<DatasetVO> children;

    @Schema(description = "数据集状态：1-启用，0-禁用", example = "1", allowableValues = {"0", "1"})
    private Integer status;

    @Schema(description = "统计信息（包含图片数量、分布等）")
    private DatasetStatistics statistics;

    @Schema(description = "图片总数（用于列表展示）", example = "100")
    private Long total;

    @Schema(description = "数据集创建时间", example = "2025-01-01T10:00:00")
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime createTime;

    @Schema(description = "数据集最后修改时间", example = "2025-01-10T15:30:00")
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime updateTime;
}

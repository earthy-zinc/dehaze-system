package com.pei.dehaze.model.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.io.Serializable;
import java.util.Map;

/**
 * 数据集统计信息VO（内部缓存使用）
 *
 * @author earthy-zinc
 * @since 2025-12-07
 */
@Data
@Schema(description = "数据集统计信息")
public class DatasetStatistics implements Serializable {

    @Schema(description = "图片总数")
    private Long imageCount;

    @Schema(description = "场景分布")
    private Map<String, Long> sceneDistribution;

    @Schema(description = "雾霾程度分布")
    private Map<String, Long> hazeDistribution;

    @Schema(description = "文件格式分布")
    private Map<String, Long> formatDistribution;
}

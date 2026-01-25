package com.pei.dehaze.model.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.io.Serializable;
import java.util.Map;

/**
 * 数据集统计信息DTO
 *
 * @author earthy-zinc
 * @since 2025-12-07
 */
@Data
@Schema(description = "数据集统计信息")
public class DatasetStatistics implements Serializable {

    @Schema(description = "数据项总数", example = "120")
    private Long itemCount;

    @Schema(description = "文件总数", example = "450")
    private Long fileCount;

    @Schema(description = "总大小（字节）", example = "15360000000")
    private Long totalSize;

    @Schema(description = "清晰图片数量", example = "120")
    private Long clearCount;

    @Schema(description = "有雾图片数量", example = "330")
    private Long hazyCount;

    @Schema(description = "场景类型分布", example = "{\"outdoor\": 280, \"indoor\": 170}")
    private Map<String, Long> sceneDistribution;

    @Schema(description = "雾霾程度分布", example = "{\"light\": 150, \"medium\": 120, \"heavy\": 60}")
    private Map<String, Long> hazeDistribution;

    @Schema(description = "文件格式分布", example = "{\"jpg\": 400, \"png\": 50}")
    private Map<String, Long> formatDistribution;
}

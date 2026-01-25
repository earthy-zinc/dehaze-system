package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * 批量删除结果
 *
 * @author earthy-zinc
 * @since 2025-01-10
 */
@Schema(description = "批量删除结果")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class BatchDeleteResult {

    @Schema(description = "总数量", example = "10")
    private Integer total;

    @Schema(description = "成功数量", example = "8")
    private Integer succeeded;

    @Schema(description = "失败数量", example = "2")
    private Integer failed;

    @Schema(description = "每个ID的处理结果")
    private List<DeleteResultItem> results;

    /**
     * 删除结果项
     */
    @Schema(description = "删除结果项")
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class DeleteResultItem {
        @Schema(description = "数据集ID", example = "1")
        private Long id;

        @Schema(description = "处理状态：success, failed", example = "success")
        private String status;

        @Schema(description = "失败原因", example = "数据集不存在")
        private String message;

        @Schema(description = "错误码", example = "RESOURCE_NOT_FOUND")
        private String errorCode;
    }
}

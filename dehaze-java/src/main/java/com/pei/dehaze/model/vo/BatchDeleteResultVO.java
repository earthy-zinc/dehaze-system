package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * 批量删除结果视图对象
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
@Data
@Schema(description = "批量删除结果视图对象")
public class BatchDeleteResultVO {

    @Schema(description = "删除成功的ID列表")
    private List<Long> successIds;

    @Schema(description = "删除失败的详细信息")
    private List<FailedItem> failedItems;

    @Schema(description = "成功删除数量")
    private Integer successCount;

    @Schema(description = "失败删除数量")
    private Integer failedCount;

    /**
     * 删除失败的详细信息
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Schema(description = "删除失败项")
    public static class FailedItem {

        @Schema(description = "图片ID")
        private Long id;

        @Schema(description = "失败原因")
        private String reason;
    }
}

package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(name = "BatchOperationResultVO", description = "批量操作结果")
public class BatchOperationResultVO {

    @Schema(description = "成功数量")
    private Integer successCount;

    @Schema(description = "失败数量")
    private Integer failedCount;

    @Schema(description = "操作消息")
    private String message;

    @Schema(description = "成功的ID列表")
    private List<Long> successIds;

    @Schema(description = "失败详情列表")
    private List<BatchActionFailureDetailVO> failureDetails;

}

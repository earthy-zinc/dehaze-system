package com.pei.dehaze.model.vo;

import com.pei.dehaze.service.importexport.model.ImportResult;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * 导入结果 VO（同步导入时返回）
 */
@Data
@Builder
@Schema(description = "导入结果")
public class ImportResultVO {

    @Schema(description = "总行数")
    private int totalRows;

    @Schema(description = "成功数")
    private int successCount;

    @Schema(description = "失败数")
    private int failureCount;

    @Schema(description = "跳过数")
    private int skippedCount;

    @Schema(description = "错误明细")
    private List<ImportResult.ImportError> errors;

    @Schema(description = "错误报告下载 URL（错误较多时生成）")
    private String errorReportUrl;

    public static ImportResultVO from(ImportResult result, String errorReportUrl) {
        return ImportResultVO.builder()
                .totalRows(result.getTotalRows())
                .successCount(result.getSuccessCount())
                .failureCount(result.getFailureCount())
                .skippedCount(result.getSkippedCount())
                .errors(result.getErrors())
                .errorReportUrl(errorReportUrl)
                .build();
    }
}

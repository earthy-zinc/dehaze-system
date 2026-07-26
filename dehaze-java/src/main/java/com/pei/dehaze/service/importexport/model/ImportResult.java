package com.pei.dehaze.service.importexport.model;

import lombok.Builder;
import lombok.Data;

import java.util.List;

/**
 * 导入结果
 */
@Data
@Builder
public class ImportResult {

    /** 总行数 */
    private int totalRows;

    /** 成功数 */
    private int successCount;

    /** 失败数 */
    private int failureCount;

    /** 跳过数（部分模式下因校验失败跳过） */
    private int skippedCount;

    /** 错误明细 */
    private List<ImportError> errors;

    /** 错误报告文件 objectName（错误较多时上传到 MinIO） */
    private String errorReportObjectName;

    @Data
    @Builder
    public static class ImportError {
        /** 行号（从 1 开始，0 表示表头错误） */
        private int row;
        /** 字段名 */
        private String field;
        /** 错误信息 */
        private String message;
    }

    public static ImportResult success(int totalRows, int successCount) {
        return ImportResult.builder()
                .totalRows(totalRows)
                .successCount(successCount)
                .failureCount(0)
                .skippedCount(0)
                .errors(List.of())
                .build();
    }

    public static ImportResult partial(int totalRows, int successCount, int failureCount, List<ImportError> errors) {
        return ImportResult.builder()
                .totalRows(totalRows)
                .successCount(successCount)
                .failureCount(failureCount)
                .skippedCount(0)
                .errors(errors == null ? List.of() : errors)
                .build();
    }
}

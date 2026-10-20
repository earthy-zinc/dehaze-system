package com.pei.clarity.service.importexport.model;

import java.util.Map;
import lombok.Data;

/**
 * 导入选项
 */
@Data
public class ImportOptions {

    /** 导入模式：all（全量）/ partial（部分） */
    private String mode;

    /** 模块特定参数（如用户导入的 deptId） */
    private Map<String, Object> extraParams;

    public static ImportOptions of(String mode) {
        ImportOptions options = new ImportOptions();
        options.mode = mode == null ? "all" : mode;
        return options;
    }

    public static ImportOptions of(String mode, Map<String, Object> extraParams) {
        ImportOptions options = of(mode);
        options.extraParams = extraParams;
        return options;
    }

    public boolean isPartialMode() {
        return "partial".equalsIgnoreCase(mode);
    }
}

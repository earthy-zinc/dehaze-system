package com.pei.dehaze.service.importexport.model;

import lombok.Builder;
import lombok.Data;

/**
 * 导出字段配置
 */
@Data
@Builder
public class ExportFieldConfig {

    /** 字段名（对应数据属性） */
    private String field;

    /** 表头标签 */
    private String label;

    /** 排序号（升序） */
    private int order;

    /** 日期格式（如 yyyy-MM-dd HH:mm:ss） */
    private String dateFormat;

    /** 字典类型（用于翻译字典值到标签） */
    private String dictType;

    /** 是否隐藏（不导出） */
    private boolean hidden;

    public static ExportFieldConfig of(String field, String label, int order) {
        return ExportFieldConfig.builder().field(field).label(label).order(order).build();
    }
}

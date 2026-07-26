package com.pei.dehaze.service.importexport.model;

import lombok.Builder;
import lombok.Data;

/**
 * 导入字段配置
 */
@Data
@Builder
public class ImportFieldConfig {

    /** 字段名（对应数据属性） */
    private String field;

    /** 表头标签（与模板表头一致） */
    private String label;

    /** 是否必填 */
    private boolean required;

    /** 日期格式（如 yyyy-MM-dd） */
    private String dateFormat;

    /** 字典类型（用于翻译标签到字典值） */
    private String dictType;

    /** 字段值正则校验 */
    private String regex;

    /** 最大长度 */
    private Integer maxLength;

    /** 默认值（导入文件未填时使用） */
    private String defaultValue;

    public static ImportFieldConfig of(String field, String label, boolean required) {
        return ImportFieldConfig.builder().field(field).label(label).required(required).build();
    }
}

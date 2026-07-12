package com.pei.dehaze.sdk.model.task;

import java.util.List;

import lombok.Data;

/**
 * 导出选项
 */
@Data
public class ExportOptions {
    /** 文件组织结构：by_item-按数据项组织, by_type-按文件类型组织 */
    private String structure = "by_item";
    /** 包含的文件类型（不传则包含所有） */
    private List<String> includeTypes;
    /** 是否包含缩略图 */
    private Boolean includeThumbnail = false;
}

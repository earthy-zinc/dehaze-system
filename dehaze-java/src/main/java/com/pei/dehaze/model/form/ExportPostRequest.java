package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;
import java.util.Map;

/**
 * POST 导出请求参数（用于复杂查询条件，请求体传递）
 */
@Data
@Schema(description = "POST 导出请求参数")
public class ExportPostRequest {

    @Schema(description = "文件格式：excel(默认) / csv", example = "excel")
    private String format;

    @Schema(description = "是否强制异步：true / false / 不传(自动判断)")
    private Boolean async;

    @Schema(description = "导出字段列表（不传则导出全部字段）", example = "[\"username\", \"nickname\"]")
    private List<String> fields;

    @Schema(description = "查询参数（各模块列表查询条件）")
    private Map<String, Object> queryParams;
}

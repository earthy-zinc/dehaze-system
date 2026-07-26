package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 导出请求参数（GET 查询参数形式）
 */
@Data
@Schema(description = "导出请求参数")
public class ExportRequest {

    @Schema(description = "文件格式：excel(默认) / csv", example = "excel")
    private String format;

    @Schema(description = "是否强制异步：true / false / 不传(自动判断)")
    private Boolean async;

    @Schema(description = "导出字段，逗号分隔（不传则导出全部字段）", example = "username,nickname,mobile")
    private String fields;

    /**
     * 将逗号分隔的 fields 字符串转为列表
     */
    public List<String> getFieldList() {
        if (fields == null || fields.isBlank()) {
            return null;
        }
        return Arrays.stream(fields.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .collect(Collectors.toList());
    }

    /**
     * 将请求参数转为查询 Map（剔除导入导出框架自身的参数）
     */
    public Map<String, Object> toQueryParams(Map<String, String[]> paramMap) {
        return paramMap.entrySet().stream()
                .filter(e -> !isFrameworkParam(e.getKey()))
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        e -> e.getValue().length == 1 ? (Object) e.getValue()[0] : e.getValue()
                ));
    }

    private boolean isFrameworkParam(String key) {
        return "format".equals(key) || "async".equals(key) || "fields".equals(key);
    }
}

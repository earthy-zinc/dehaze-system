package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.HashMap;
import java.util.Map;

/**
 * 导入请求参数
 */
@Data
@Schema(description = "导入请求参数")
public class ImportRequest {

    @Schema(description = "导入模式：all(全量,默认) / partial(部分)")
    private String mode;

    @Schema(description = "是否强制异步：true / false / 不传(自动判断)")
    private Boolean async;

    public String getModeOrDefault() {
        return mode == null || mode.isBlank() ? "all" : mode;
    }

    /**
     * 从 HTTP 参数表中提取模块特定参数（剔除框架自身的 file/mode/async）
     */
    public Map<String, Object> toExtraParams(Map<String, String[]> paramMap) {
        Map<String, Object> extraParams = new HashMap<>();
        paramMap.forEach((key, values) -> {
            if (!"file".equals(key) && !"mode".equals(key) && !"async".equals(key)) {
                extraParams.put(key, values.length == 1 ? values[0] : values);
            }
        });
        return extraParams;
    }
}

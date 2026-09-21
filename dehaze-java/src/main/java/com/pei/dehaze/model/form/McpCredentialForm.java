package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.util.Map;

/** 外部服务凭据配置表单（加密存储，仅录入/更新，不回显明文） */
@Schema(description = "MCP Server 凭据配置表单")
@Data
public class McpCredentialForm {

    @Schema(description = "API Key 等外部服务凭据")
    @Size(max = 1024)
    private String apiKey;

    @Schema(description = "其他凭据字段(服务层加密存储)")
    private Map<String, String> extra;

    @Schema(description = "清除已配置凭据(凭据轮换/吊销场景)")
    private Boolean clear = false;
}

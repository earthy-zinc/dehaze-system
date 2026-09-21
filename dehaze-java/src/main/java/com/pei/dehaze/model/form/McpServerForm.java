package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Schema(description = "外部 MCP Server 注册表单")
@Data
public class McpServerForm {

    @NotBlank(message = "Server 名称不能为空")
    @Size(min = 1, max = 128)
    private String name;

    @Size(max = 512)
    private String description;

    @Schema(description = "传输协议(streamable-http/sse)")
    private String protocolType = "streamable-http";

    @Size(max = 512)
    private String endpoint;

    @Schema(description = "鉴权方式(none/api_key/oauth2)")
    @Size(max = 32)
    private String authType;
}

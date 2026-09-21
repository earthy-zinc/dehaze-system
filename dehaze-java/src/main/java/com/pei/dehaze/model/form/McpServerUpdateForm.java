package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Size;
import lombok.Data;

/** 外部 MCP Server 更新表单（null 视为不修改） */
@Schema(description = "外部 MCP Server 更新表单")
@Data
public class McpServerUpdateForm {

    @Size(min = 1, max = 128)
    private String name;

    @Size(max = 512)
    private String description;

    private String protocolType;

    @Size(max = 512)
    private String endpoint;

    @Size(max = 32)
    private String authType;
}

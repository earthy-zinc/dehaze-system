package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;

/** 命名空间项（工具分组）：name 为命名空间标识，toolNames 为组内工具名数组 */
@Schema(description = "MCP 命名空间视图")
@Data
public class McpNamespaceVO {

    private String name;

    private List<String> toolNames;
}

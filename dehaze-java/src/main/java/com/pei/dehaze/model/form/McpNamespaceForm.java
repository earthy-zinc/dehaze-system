package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

/** 命名空间配置项（覆盖式更新，整组提交） */
@Schema(description = "MCP 命名空间配置项")
@Data
public class McpNamespaceForm {

    @Schema(description = "命名空间标识(工具分组)")
    private String name;

    @Schema(description = "组内工具名数组")
    private List<String> toolNames = new ArrayList<>();
}

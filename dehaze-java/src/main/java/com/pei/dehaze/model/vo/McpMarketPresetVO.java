package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;

@Schema(description = "MCP 市场预设目录项")
@Data
public class McpMarketPresetVO {

    private String presetId;

    private String name;

    private String description;

    private List<String> capabilityTags;

    private Boolean installed;
}

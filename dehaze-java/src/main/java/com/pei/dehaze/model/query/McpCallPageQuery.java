package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Schema(description = "外部 MCP 工具调用审计查询参数")
@Data
@EqualsAndHashCode(callSuper = true)
public class McpCallPageQuery extends BasePageQuery {

    @Schema(description = "按 Server 筛选")
    private Long serverId;

    @Schema(description = "按工具名筛选")
    private String toolName;
}

package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Schema(description = "外部 MCP Server 分页查询参数")
@Data
@EqualsAndHashCode(callSuper = true)
public class McpServerPageQuery extends BasePageQuery {

    @Schema(description = "关键字(按名称/描述模糊搜索)")
    private String keyword;

    @Schema(description = "状态筛选(1:启用;0:禁用)")
    private Integer status;
}

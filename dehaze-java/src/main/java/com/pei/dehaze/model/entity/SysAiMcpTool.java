package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/** 外部 MCP Server 工具清单（表 sys_ai_mcp_tool），随每次拉取覆盖式重建 */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysAiMcpTool extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long serverId;

    private String name;

    private String description;

    /** 参数 schema 概要（JSON 文本） */
    private String inputSchema;
}

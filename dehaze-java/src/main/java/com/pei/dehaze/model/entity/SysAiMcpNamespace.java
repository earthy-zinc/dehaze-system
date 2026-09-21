package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/** 外部 MCP Server 命名空间配置（表 sys_ai_mcp_namespace），整组覆盖式更新 */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysAiMcpNamespace extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long serverId;

    private String namespace;

    /** 组内工具名数组（JSON 文本） */
    private String toolNames;
}

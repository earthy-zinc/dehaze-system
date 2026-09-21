package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;

/**
 * 外部 MCP Server 注册表（表 sys_ai_mcp_server）。
 *
 * <p>credentials 为 AES 密文 JSON（{"api_key":"<密文>","extra":{"k":"<密文>"}}），
 * 仅录入/更新，不回显明文、不暴露给 LLM。
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysAiMcpServer extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String name;

    private String description;

    /** streamable-http / sse（stdio 无网络端点，不支持） */
    private String protocolType;

    private String endpoint;

    private String authType;

    private String credentials;

    /** online / offline */
    private String health;

    private LocalDateTime lastCheckTime;

    private Integer status;

    /** 冗余工具数量，避免列表逐条子查询 */
    private Integer toolCount;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;
}

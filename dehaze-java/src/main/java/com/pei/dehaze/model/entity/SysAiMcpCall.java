package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;

import java.time.LocalDateTime;

/**
 * 外部 MCP 工具调用审计（表 sys_ai_mcp_call，只追加不删除）。
 *
 * <p>表无 update_time/create_by，故不继承 BaseEntity。
 */
@Data
public class SysAiMcpCall {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private Long serverId;

    /** Server 名称冗余快照，避免审计时关联已软删的 Server */
    private String serverName;

    private String toolName;

    private String request;

    private String response;

    private Integer status;

    /** success / failure */
    private String result;

    private Integer latencyMs;

    @TableField(fill = FieldFill.INSERT)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}

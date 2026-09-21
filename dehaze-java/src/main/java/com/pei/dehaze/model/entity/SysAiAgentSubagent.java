package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.io.Serial;
import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * 智能体-子 Agent 关联（覆盖式维护）
 *
 * @author dehaze
 */
@Data
@TableName("sys_ai_agent_subagent")
public class SysAiAgentSubagent implements Serializable {

    @TableId(value = "parent_agent_id", type = IdType.INPUT)
    private Long parentAgentId;

    private Long subagentAgentId;

    private Long endpointId;

    private Integer priority;

    private LocalDateTime createTime;

    @Serial
    private static final long serialVersionUID = 1L;
}

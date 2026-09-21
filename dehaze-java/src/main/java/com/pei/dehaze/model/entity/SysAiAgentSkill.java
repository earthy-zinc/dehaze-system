package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.io.Serial;
import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * 智能体-Skill 关联（覆盖式维护）
 *
 * @author dehaze
 */
@Data
@TableName("sys_ai_agent_skill")
public class SysAiAgentSkill implements Serializable {

    @TableId(value = "agent_id", type = IdType.INPUT)
    private Long agentId;

    private String skillName;

    private LocalDateTime createTime;

    @Serial
    private static final long serialVersionUID = 1L;
}

package com.pei.clarity.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.io.Serial;
import java.io.Serializable;
import java.time.LocalDateTime;
import lombok.Data;

/**
 * 智能体-Skill 关联（覆盖式维护）
 *
 * @author clarity
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

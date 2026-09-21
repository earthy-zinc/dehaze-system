package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler;
import lombok.Data;

import java.io.Serial;
import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.Map;

/**
 * AI 智能体版本快照（只追加，无逻辑删除）
 *
 * @author dehaze
 */
@Data
@TableName(value = "sys_ai_agent_version", autoResultMap = true)
public class SysAiAgentVersion implements Serializable {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long agentId;

    private Integer versionNo;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private Map<String, Object> snapshot;

    private Integer status;

    private String changeNote;

    private Long operatorId;

    private LocalDateTime createTime;

    @Serial
    private static final long serialVersionUID = 1L;
}

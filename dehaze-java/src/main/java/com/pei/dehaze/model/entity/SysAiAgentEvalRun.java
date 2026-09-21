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
import java.util.List;
import java.util.Map;

/**
 * 智能体评测执行记录（只追加）
 *
 * @author dehaze
 */
@Data
@TableName(value = "sys_ai_agent_eval_run", autoResultMap = true)
public class SysAiAgentEvalRun implements Serializable {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long agentId;

    private Long datasetId;

    private String triggerType;

    private Integer status;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private Map<String, Object> scoreSummary;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private List<Map<String, Object>> results;

    private Long createBy;

    private LocalDateTime createTime;

    private LocalDateTime updateTime;

    @Serial
    private static final long serialVersionUID = 1L;
}

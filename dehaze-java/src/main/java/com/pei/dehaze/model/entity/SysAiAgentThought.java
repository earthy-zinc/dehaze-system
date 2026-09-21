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
 * 智能体推理步骤（消息详情的思考链）
 *
 * @author dehaze
 */
@Data
@TableName(value = "sys_ai_agent_thought", autoResultMap = true)
public class SysAiAgentThought implements Serializable {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long messageId;

    private Long conversationId;

    private Integer position;

    private String agentCode;

    private Integer isSubagent;

    private String thought;

    private String tool;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private Map<String, Object> toolInput;

    private String observation;

    private String summary;

    private Integer status;

    private Integer latencyMs;

    private String error;

    private LocalDateTime createTime;

    @Serial
    private static final long serialVersionUID = 1L;
}

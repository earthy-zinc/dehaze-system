package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serial;
import java.util.Map;

/**
 * 外部 A2A 智能体端点
 *
 * @author dehaze
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName(value = "sys_ai_agent_endpoint", autoResultMap = true)
public class SysAiAgentEndpoint extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String name;

    private String agentCardUrl;

    private String baseUrl;

    private String authType;

    private String credential;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private Map<String, Object> agentCard;

    private Integer status;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}

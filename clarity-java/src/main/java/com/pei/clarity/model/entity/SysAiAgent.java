package com.pei.clarity.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler;
import com.pei.clarity.common.base.BaseEntity;
import java.io.Serial;
import java.util.List;
import java.util.Map;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * AI 智能体
 *
 * @author clarity
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName(value = "sys_ai_agent", autoResultMap = true)
public class SysAiAgent extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String agentCode;

    private String name;

    private String description;

    private String systemPrompt;

    private String modelId;

    private String reasoningMode;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private Map<String, Object> config;

    private Integer isSubagent;

    private Integer isTeam;

    private Integer isExposed;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private List<Map<String, Object>> permissions;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private List<String> tags;

    private Integer sortOrder;

    private Integer status;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}

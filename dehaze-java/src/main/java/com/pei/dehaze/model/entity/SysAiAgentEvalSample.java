package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serial;
import java.util.List;

/**
 * 智能体评测样本（无逻辑删除，随评测集物理清理）
 *
 * @author dehaze
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName(value = "sys_ai_agent_eval_sample", autoResultMap = true)
public class SysAiAgentEvalSample extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long datasetId;

    private String taskGoal;

    private String allowedInput;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private List<String> tools;

    private String expectedProcess;

    private String expectedResult;

    private String forbiddenBehavior;

    private String riskLevel;

    @Serial
    private static final long serialVersionUID = 1L;
}

package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serial;

/**
 * 智能体评测集
 *
 * @author dehaze
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_ai_agent_eval_dataset")
public class SysAiAgentEvalDataset extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long agentId;

    private String name;

    private String description;

    private String datasetType;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}

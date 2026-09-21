package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serial;

/**
 * 人工复核项
 *
 * @author dehaze
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_ai_eval_review")
public class SysAiEvalReview extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long runId;

    private Long sampleId;

    private Long agentId;

    private Integer judgePassed;

    private String riskLevel;

    private Integer status;

    private Integer agree;

    private Long reviewerId;

    private String remark;

    @Serial
    private static final long serialVersionUID = 1L;
}

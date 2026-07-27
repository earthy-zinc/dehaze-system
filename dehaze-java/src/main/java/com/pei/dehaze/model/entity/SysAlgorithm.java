package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.pei.dehaze.common.base.BaseEntity;
import com.pei.dehaze.common.enums.AlgorithmStatusEnum;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;

@Data
@EqualsAndHashCode(callSuper = false)
public class SysAlgorithm extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long parentId;

    private String type;

    private String name;

    private String path;

    private String size;

    private String img;

    private String params;

    private String flops;

    private String importPath;

    private String description;

    /**
     * 算法版本号（语义化版本，如 1.0.0）
     */
    private String version;

    /**
     * 算法状态：1-草稿, 2-测试中, 3-待审核, 4-已发布, 5-已停用, 6-已归档
     */
    private Integer status;

    /**
     * 审核人ID
     */
    private Long auditBy;

    /**
     * 审核时间
     */
    private LocalDateTime auditTime;

    /**
     * 审核备注（驳回时必填）
     */
    private String auditRemark;

    /**
     * 逻辑删除标识(0:未删除;1:已删除)
     */
    @TableLogic
    private Integer deleted;

    /**
     * 是否为终态（已发布/已停用/已归档）
     */
    public boolean isFinalStatus() {
        return status != null && AlgorithmStatusEnum.FINAL_STATUSES.contains(status);
    }

    /**
     * 是否可编辑
     */
    public boolean isEditable() {
        return status != null && AlgorithmStatusEnum.EDITABLE_STATUSES.contains(status);
    }
}

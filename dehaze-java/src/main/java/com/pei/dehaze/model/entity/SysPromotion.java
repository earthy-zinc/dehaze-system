package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serial;
import java.time.LocalDateTime;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_promotion")
public class SysPromotion extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String name;

    private String type;

    private String description;

    private LocalDateTime startTime;

    private LocalDateTime endTime;

    private String activityRules;

    private String applicableScope;

    private Integer newUserOnly;

    private Integer status;

    @TableLogic
    private Integer deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}

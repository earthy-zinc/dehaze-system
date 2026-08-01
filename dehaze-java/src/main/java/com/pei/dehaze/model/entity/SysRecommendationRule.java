package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.dehaze.common.base.BaseEntity;
import com.pei.dehaze.common.handler.LongListTypeHandler;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serial;
import java.util.List;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName(value = "sys_recommendation_rule", autoResultMap = true)
public class SysRecommendationRule extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String ruleName;

    private String sceneType;

    @TableField(typeHandler = LongListTypeHandler.class)
    private List<Long> algorithmIds;

    private Integer weight;

    private Integer enabled;

    @TableLogic
    private Integer deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}

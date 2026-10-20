package com.pei.clarity.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.clarity.common.base.BaseEntity;
import com.pei.clarity.common.handler.StringListTypeHandler;
import java.io.Serial;
import java.util.List;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName(value = "sys_recommendation_rule", autoResultMap = true)
public class SysRecommendationRule extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String ruleName;

    private String sceneType;

    @TableField(typeHandler = StringListTypeHandler.class)
    private List<String> modelIds;

    private Integer weight;

    private Integer enabled;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}

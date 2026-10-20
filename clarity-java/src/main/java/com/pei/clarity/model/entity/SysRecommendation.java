package com.pei.clarity.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler;
import com.pei.clarity.common.base.BaseEntity;
import java.io.Serial;
import java.util.List;
import java.util.Map;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName(value = "sys_recommendation", autoResultMap = true)
public class SysRecommendation extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private String imageMd5;

    private String targetType;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private List<Map<String, Object>> topModels;

    @TableField(typeHandler = JacksonTypeHandler.class)
    private Map<String, Object> analysisResult;

    private Integer feedback;

    private String adoptedModelId;

    @Serial
    private static final long serialVersionUID = 1L;
}

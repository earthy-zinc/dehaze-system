package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serial;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_message_template")
public class SysMessageTemplate extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String code;

    private String name;

    private String type;

    private String titleTemplate;

    private String contentTemplate;

    private Integer priority;

    private String channels;

    private String variables;

    private Integer status;

    @TableLogic
    private Integer deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}

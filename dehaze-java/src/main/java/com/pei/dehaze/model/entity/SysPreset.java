package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_preset")
public class SysPreset extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String name;

    private String type;

    private Long algorithmId;

    private String params;

    private Long userId;

    private Integer isDefault;
}

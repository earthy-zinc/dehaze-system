package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
public class SysEvalLog extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long algorithmId;

    private Long predFileId;

    private String predMd5;

    private String predUrl;

    private Long gtFileId;

    private String gtMd5;

    private String gtUrl;

    private Integer time;

    private String result;

    @TableField(fill = FieldFill.INSERT)
    private Long createBy;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private Long updateBy;
}

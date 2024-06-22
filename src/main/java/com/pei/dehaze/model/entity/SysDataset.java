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
public class SysDataset  extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long parentId;

    private String type;

    private String name;

    private String description;

    private String path;

    private String size;

    private Integer total;

    private Integer status;

    private Integer deleted;

    @TableField(fill = FieldFill.INSERT)
    private Long createBy;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private Long updateBy;

}

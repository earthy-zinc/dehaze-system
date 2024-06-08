package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;

@Data
public class SysAlgorithm extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long parentId;

    private String type;

    private String name;

    private String path;

    private Long size;

    private String importPath;

    private String description;

    private Integer status;

    @TableField("create_by")
    private Long createBy;

    @TableField("update_by")
    private Long updateBy;
}

package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;

@Data
public class SysImage extends BaseEntity {
    @TableId(type = IdType.AUTO)
    private Long id;

    private String type;

    private String url;

    private String name;

    private String resolution;

    private Integer size;

    private String extendName;

    private String path;

    private String md5;
}

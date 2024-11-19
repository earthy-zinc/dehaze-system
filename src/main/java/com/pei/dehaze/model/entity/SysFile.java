package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
@Builder
public class SysFile extends BaseEntity {
    @TableId(type = IdType.AUTO)
    private Long id;

    private String type;

    private String url;

    private String name;

    private String objectName;

    private String size;

    private String path;

    private String md5;
}

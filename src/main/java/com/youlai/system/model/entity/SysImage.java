package com.youlai.system.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.youlai.system.common.base.BaseEntity;
import lombok.Data;

@Data
public class SysImage  extends BaseEntity {
    @TableId(type = IdType.AUTO)
    private Long id;

}

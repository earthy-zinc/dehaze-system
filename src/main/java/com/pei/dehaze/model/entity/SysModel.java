package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;

@Data
public class SysModel  extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

}

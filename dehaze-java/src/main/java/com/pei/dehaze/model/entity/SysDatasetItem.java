package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import lombok.Data;

@Data
public class SysDatasetItem {
    @TableId(type = IdType.AUTO)
    private Long id;

    private Long datasetId;

    private String name;
}

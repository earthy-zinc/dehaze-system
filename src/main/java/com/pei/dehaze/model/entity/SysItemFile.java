package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import lombok.Data;

@Data
public class SysItemFile {
    @TableId(type = IdType.AUTO)
    private Long id;

    private Long itemId;

    private Long fileId;

    private Long thumbnailFileId;

    private String type;

    private String description;
}

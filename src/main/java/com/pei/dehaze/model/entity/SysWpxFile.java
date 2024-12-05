package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode
public class SysWpxFile {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long originFileId;

    private String originMd5;

    private String originPath;

    private Long newFileId;

    private String newPath;

    private String newMd5;
}

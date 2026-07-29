package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;

@Data
@EqualsAndHashCode(callSuper = false)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SysFile extends BaseEntity {
    @TableId(type = IdType.AUTO)
    private Long id;

    private String type;

    private String url;

    private String name;

    private String objectName;

    /**
     * 文件大小（格式化显示，如 "2.44MB"）
     */
    private String size;

    private String path;

    private String md5;

    private Integer deleted;
}

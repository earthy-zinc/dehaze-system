package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serial;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_favorite")
public class SysFavorite extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private String targetType;

    private Long targetId;

    /**
     * 收藏对象是否已失效(0:正常;1:已失效)
     */
    private Integer isInvalid;

    @TableLogic
    private Integer deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}

package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.pei.dehaze.common.base.BaseEntity;
import com.pei.dehaze.common.enums.StatusEnum;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
public class SysDataset  extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long parentId;

    private String type;

    private String name;

    private String img;

    private String description;

    private String path;

    private String size;

    /**
     * 状态(1:正常;0:禁用)
     */
    private StatusEnum status;

    /**
     * 逻辑删除标识(1:已删除;0:未删除)
     */
    @TableLogic
    private Integer deleted;

    /**
     * 使用次数（用户使用该数据集的次数）
     */
    private Long usageCount;

}

package com.pei.clarity.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.clarity.common.base.BaseEntity;
import java.io.Serial;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_promotion_package")
public class SysPromotionPackage extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long promotionId;

    private Long packageId;

    private String discountType;

    private Long discountValue;

    @Serial
    private static final long serialVersionUID = 1L;
}

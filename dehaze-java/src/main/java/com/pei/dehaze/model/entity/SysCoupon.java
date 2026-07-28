package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serial;
import java.time.LocalDateTime;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_coupon")
public class SysCoupon extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String name;

    private String type;

    private Long faceValue;

    private Long threshold;

    private String validType;

    private LocalDateTime validStart;

    private LocalDateTime validEnd;

    private Integer validDays;

    private Integer totalQty;

    private Integer issuedQty;

    private Integer usedQty;

    private Integer perUserLimit;

    private String applicableScope;

    private Integer status;

    @TableLogic
    private Integer deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}

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
@TableName("sys_order")
public class SysOrder extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String orderNo;

    private Long userId;

    private Long packageId;

    private String packageName;

    private String packageLevel;

    private Integer periodDays;

    private Long originalPrice;

    private Long discountAmount;

    private Long couponId;

    private Long couponAmount;

    private Long payableAmount;

    private Long paidAmount;

    private String payMethod;

    private Integer status;

    private LocalDateTime expireTime;

    private LocalDateTime effectiveTime;

    private LocalDateTime packageExpireTime;

    private LocalDateTime paidTime;

    private String cancelReason;

    private Integer isAutoRenew;

    @TableLogic
    private Integer deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}

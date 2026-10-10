package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.dehaze.common.base.BaseEntity;
import java.io.Serial;
import java.time.LocalDateTime;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_user_coupon")
public class SysUserCoupon extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private Long couponId;

    private Integer status;

    private LocalDateTime receiveTime;

    private LocalDateTime expireTime;

    private LocalDateTime usedTime;

    private Long usedOrderId;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}

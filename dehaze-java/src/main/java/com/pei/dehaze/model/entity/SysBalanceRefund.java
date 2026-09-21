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
@TableName("sys_balance_refund")
public class SysBalanceRefund extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String refundNo;

    private Long userId;

    private Long amount;

    private Integer status;

    private String channel;

    private String channelRefundNo;

    private LocalDateTime applyTime;

    private LocalDateTime auditTime;

    private Long auditorId;

    private String auditRemark;

    private LocalDateTime refundTime;

    private String errorMessage;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}

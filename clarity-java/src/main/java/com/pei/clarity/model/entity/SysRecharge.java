package com.pei.clarity.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.baomidou.mybatisplus.annotation.TableName;
import com.pei.clarity.common.base.BaseEntity;
import java.io.Serial;
import java.time.LocalDateTime;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName("sys_recharge")
public class SysRecharge extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String rechargeNo;

    private Long userId;

    private Long amount;

    private String payMethod;

    private Integer status;

    private String channelPaymentNo;

    private LocalDateTime payTime;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;

    @Serial
    private static final long serialVersionUID = 1L;
}

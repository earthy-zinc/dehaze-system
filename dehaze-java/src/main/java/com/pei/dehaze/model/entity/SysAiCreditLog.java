package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/** 积分余额变动流水（表 sys_ai_credit_log，只追加） */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysAiCreditLog extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    /** recharge/vip_gift/trial/admin_adjust/refund/consume/vip_gift_expire */
    private String source;

    /** 正数增加、负数扣减 */
    private Long amount;

    private Long balanceAfter;

    private Long relatedId;

    private String reason;

    /** 人工调整/客服补偿的操作人，系统自动为 null */
    private Long operatorId;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;
}

package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.math.BigDecimal;

/** 用户售价档位明细（表 sys_ai_model_price_detail）：token 类型 × 时段 × 上下文分段 */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysAiModelPriceDetail extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long priceId;

    /** input/cached/output */
    private String tokenType;

    /** peak/idle */
    private String timeSlot;

    private Long minTokens;

    private Long maxTokens;

    private BigDecimal unitPrice;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;
}

package com.pei.clarity.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import com.pei.clarity.common.base.BaseEntity;
import java.math.BigDecimal;
import lombok.Data;
import lombok.EqualsAndHashCode;

/** 用户售价档位明细（表 sys_ai_model_price_detail）：token 类型 × 时段 × 上下文分段 */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysAiModelPriceDetail extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long priceId;

    /** input/cached/output */
    private String tokenType;

    /** 计量种类(token:token计量;unit:按 billing_unit 单位计量，非 token 模型单行 unit 档) */
    private String usageKind;

    /** peak/idle */
    private String timeSlot;

    private Long minTokens;

    private Long maxTokens;

    private BigDecimal unitPrice;

    @TableLogic(value = "0", delval = "id")
    private Long deleted;
}
